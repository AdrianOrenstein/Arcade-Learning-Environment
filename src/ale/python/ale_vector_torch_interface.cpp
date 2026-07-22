// PyTorch C++ backend for the ALE vector env.
//
// Mirrors the Python custom ops in _torch_ops.py, but registers
// ale_cpp::send / ale_cpp::recv directly with the PyTorch dispatcher so step
// latency avoids Python op-dispatch overhead. It lives in its own op namespace
// (ale_cpp) and coexists with the pure-Python ale::send / ale::recv ops, so
// AtariVectorEnv.torch(cpp=True) can be benchmarked against torch(cpp=False).
//
// On CUDA the send is fully asynchronous: the action ids are copied D2H into a
// pinned buffer on the current stream and the blocking vectorizer->send() is
// deferred to a cudaLaunchHostFunc callback, so the op returns immediately and
// overlaps with whatever the caller does next. recv() synchronises the stream
// (so the send callback has run) before draining the workers and copying the
// results back H2D.
//
// Supports both the discrete action space (paddle strength fixed at 1.0) and the
// continuous one (polar action mapped to a discrete id + paddle strength on the
// host, matching AtariVectorEnv.send()).

#include <torch/library.h>
#include <ATen/ATen.h>

#include "ale/vector/env_vectorizer.hpp"

#include <nanobind/nanobind.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifdef BUILD_VECTOR_TORCH_CUDA
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#ifndef _WIN32
#include <dlfcn.h>
#endif
#endif

namespace nb = nanobind;

using ale::vector::Action;
using ale::vector::BatchResult;
using ale::vector::EnvVectorizer;

namespace {

using RecvResult = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                              at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
using RecvResult9 = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                               at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                               at::Tensor>;

/// Per-env state shared between op calls. Registered from Python during torch()
/// setup and looked up by handle id (the Python id() of the env).
struct TorchHandle {
    EnvVectorizer* vectorizer;
    void* pinned_in;             // persistent pinned D2H target for CUDA send:
                                 //   int32[batch] discrete, float[batch*3] continuous
    int64_t batch_size;
    bool is_cuda;                // whether step results should land on CUDA
    bool continuous;             // continuous action space (polar -> discrete map)
    float threshold;             // continuous_action_threshold
    int32_t map_action_idx[18];  // (3,3,2) flattened; continuous only
    bool send_failed;            // set by the CUDA send callback if send() threw
};

std::unordered_map<int64_t, TorchHandle> g_handles;

TorchHandle& get_handle(int64_t handle_id) {
    auto it = g_handles.find(handle_id);
    if (it == g_handles.end()) {
        throw std::runtime_error("ale_cpp: unknown torch handle " +
                                 std::to_string(handle_id));
    }
    return it->second;
}

/// Map one continuous action (radius, theta, fire) to a discrete action id and
/// paddle strength, mirroring AtariVectorEnv.send()'s polar discretization.
inline void continuous_to_action(const TorchHandle& d, const float* a3,
                                 int32_t& action_id, float& paddle) {
    const float r = a3[0], theta = a3[1], fire = a3[2], t = d.threshold;
    const float x = r * std::cos(theta);
    const float y = r * std::sin(theta);
    const int h = (x > t) - (x < -t) + 1;   // 0..2
    const int v = (y > t) - (y < -t) + 1;   // 0..2
    const int f = (fire > t) ? 1 : 0;       // 0..1
    action_id = d.map_action_idx[h * 6 + v * 2 + f];
    paddle = r;
}

/// Build the Action batch from the raw action buffer (int32 ids for discrete,
/// float (batch,3) for continuous) and dispatch it to the vectorizer. With
/// allow_pending_action the extra last id is the pending action; the
/// vectorizer detects it and holds the env.
void build_and_send(const TorchHandle& d, const void* raw) {
    std::vector<Action> actions(static_cast<std::size_t>(d.batch_size));
    if (d.continuous) {
        const float* a = static_cast<const float*>(raw);
        for (int64_t i = 0; i < d.batch_size; ++i) {
            int32_t aid;
            float paddle;
            continuous_to_action(d, a + i * 3, aid, paddle);
            actions[i] = {static_cast<int>(i), aid, paddle, false};
        }
    } else {
        const int32_t* a = static_cast<const int32_t*>(raw);
        for (int64_t i = 0; i < d.batch_size; ++i) {
            actions[i] = {static_cast<int>(i), a[i], 1.0f, false};
        }
    }
    d.vectorizer->send(actions);
}

std::vector<int64_t> obs_shape_for(EnvVectorizer* vec, int64_t batch) {
    auto [stack, height, width, channels] = vec->observation_shape();
    if (vec->is_grayscale()) {
        return {batch, stack, height, width};
    }
    return {batch, stack, height, width, 3};
}

/// Coerce an action tensor to contiguous int32 (no-op when already int32).
at::Tensor as_int32(const at::Tensor& actions) {
    at::Tensor a = actions.scalar_type() == at::kInt ? actions : actions.to(at::kInt);
    return a.contiguous();
}

// -------------------------------------------------------------------- CPU path

at::Tensor send_cpu(TorchHandle& d, const at::Tensor& actions) {
    if (d.continuous) {
        at::Tensor a = actions.to(at::kFloat).contiguous();
        build_and_send(d, a.data_ptr<float>());
    } else {
        at::Tensor a = as_int32(actions);
        build_and_send(d, a.data_ptr<int32_t>());
    }
    return actions.new_empty({0});
}

// Returns the 8 core step tensors, plus final_obs as a 9th when with_final
// (SameStep autoreset mode); final_obs is valid only for envs that are done.
std::vector<at::Tensor> recv_cpu(TorchHandle& d, bool with_final) {
    BatchResult result = d.vectorizer->recv();
    const int64_t batch = static_cast<int64_t>(result.batch_size());
    const int64_t obs_bytes = batch * static_cast<int64_t>(d.vectorizer->stacked_obs_size());

    auto i32 = at::TensorOptions().dtype(at::kInt);
    auto obs_shape = obs_shape_for(d.vectorizer, batch);
    at::Tensor obs = at::empty(obs_shape, at::TensorOptions().dtype(at::kByte));
    at::Tensor reward = at::empty({batch}, at::TensorOptions().dtype(at::kFloat));
    at::Tensor term = at::empty({batch}, at::TensorOptions().dtype(at::kBool));
    at::Tensor trunc = at::empty({batch}, at::TensorOptions().dtype(at::kBool));
    at::Tensor env_id = at::empty({batch}, i32);
    at::Tensor lives = at::empty({batch}, i32);
    at::Tensor frame = at::empty({batch}, i32);
    at::Tensor ep_frame = at::empty({batch}, i32);

    std::memcpy(obs.data_ptr(), result.obs_data(), obs_bytes);
    std::memcpy(reward.data_ptr(), result.rewards_data(), batch * sizeof(float));
    std::memcpy(term.data_ptr(), result.terminations_data(), batch * sizeof(bool));
    std::memcpy(trunc.data_ptr(), result.truncations_data(), batch * sizeof(bool));
    std::memcpy(env_id.data_ptr(), result.env_ids_data(), batch * sizeof(int32_t));
    std::memcpy(lives.data_ptr(), result.lives_data(), batch * sizeof(int32_t));
    std::memcpy(frame.data_ptr(), result.frame_numbers_data(), batch * sizeof(int32_t));
    std::memcpy(ep_frame.data_ptr(), result.episode_frame_numbers_data(),
                batch * sizeof(int32_t));

    std::vector<at::Tensor> out = {obs, reward, term, trunc,
                                   env_id, lives, frame, ep_frame};
    if (with_final) {
        at::Tensor final_obs = at::empty(obs_shape, at::TensorOptions().dtype(at::kByte));
        if (result.has_final_obs()) {
            std::memcpy(final_obs.data_ptr(), result.final_obs_data(), obs_bytes);
        } else {
            final_obs.zero_();
        }
        out.push_back(final_obs);
    }
    return out;
}

// ------------------------------------------------------------------- CUDA path
#ifdef BUILD_VECTOR_TORCH_CUDA

// cudart is loaded lazily via dlsym (RTLD_NOLOAD) so this module does not
// hard-link a libcudart that might differ from the one PyTorch already loaded.
using fn_memcpy_async_t = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
using fn_stream_sync_t = cudaError_t (*)(cudaStream_t);
using fn_launch_host_t = cudaError_t (*)(cudaStream_t, cudaHostFn_t, void*);
using fn_error_string_t = const char* (*)(cudaError_t);

fn_memcpy_async_t p_cudaMemcpyAsync = nullptr;
fn_stream_sync_t p_cudaStreamSynchronize = nullptr;
fn_launch_host_t p_cudaLaunchHostFunc = nullptr;
fn_error_string_t p_cudaGetErrorString = nullptr;
bool g_cuda_loaded = false;

bool load_cuda_fns() {
#ifdef _WIN32
    return false;  // CUDA torch backend is Linux-only for now
#else
    void* h = dlopen("libcudart.so.13", RTLD_LAZY | RTLD_NOLOAD);
    if (!h) h = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_NOLOAD);
    if (!h) h = dlopen("libcudart.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!h) h = dlopen("libcudart.so.13", RTLD_LAZY | RTLD_GLOBAL);
    if (!h) return false;
    p_cudaMemcpyAsync = (fn_memcpy_async_t)dlsym(h, "cudaMemcpyAsync");
    p_cudaStreamSynchronize = (fn_stream_sync_t)dlsym(h, "cudaStreamSynchronize");
    p_cudaLaunchHostFunc = (fn_launch_host_t)dlsym(h, "cudaLaunchHostFunc");
    p_cudaGetErrorString = (fn_error_string_t)dlsym(h, "cudaGetErrorString");
    return p_cudaMemcpyAsync && p_cudaStreamSynchronize && p_cudaLaunchHostFunc &&
           p_cudaGetErrorString;
#endif
}

void ensure_cuda() {
    if (!g_cuda_loaded) {
        if (!load_cuda_fns()) {
            throw std::runtime_error(
                "ale_cpp: could not load libcudart symbols; is PyTorch CUDA loaded?");
        }
        g_cuda_loaded = true;
    }
}

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("ale_cpp CUDA error (") + what +
                                 "): " + p_cudaGetErrorString(e));
    }
}

// Runs on a CUDA driver thread once the D2H of the action ids has completed.
// Must not call into the CUDA API. Exceptions cannot propagate from here; a bad
// action would otherwise have been rejected synchronously on the CPU path.
void send_host_callback(void* arg) {
    auto* d = static_cast<TorchHandle*>(arg);
    try {
        build_and_send(*d, d->pinned_in);
    } catch (...) {
        // Exceptions can't propagate from a driver callback; flag it so the
        // matching recv() throws instead of blocking forever on results that
        // were never requested.
        d->send_failed = true;
    }
}

at::Tensor send_cuda(TorchHandle& d, const at::Tensor& actions) {
    ensure_cuda();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Async D2H of the raw actions into the persistent pinned buffer (int32[batch]
    // for discrete, float[batch*3] for continuous), then defer the blocking send()
    // to fire when the copy completes. `a` stays valid until the copy runs by CUDA
    // stream ordering (its memory is only reused after this stream work). The
    // send/recv usage is strictly alternating, so the pinned buffer is consumed by
    // the callback before the next send overwrites it.
    at::Tensor a;
    size_t bytes;
    const void* src;
    if (d.continuous) {
        a = actions.to(at::kFloat).contiguous();
        bytes = d.batch_size * 3 * sizeof(float);
        src = a.data_ptr<float>();
    } else {
        a = as_int32(actions);
        bytes = d.batch_size * sizeof(int32_t);
        src = a.data_ptr<int32_t>();
    }
    cuda_check(p_cudaMemcpyAsync(d.pinned_in, src, bytes,
                                 cudaMemcpyDeviceToHost, stream),
               "send: D2H actions");
    cuda_check(p_cudaLaunchHostFunc(stream, &send_host_callback, &d),
               "send: launch host func");
    return actions.new_empty({0});
}

std::vector<at::Tensor> recv_cuda(TorchHandle& d, bool with_final) {
    ensure_cuda();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Wait for the deferred send() callback to have run before draining workers.
    cuda_check(p_cudaStreamSynchronize(stream), "recv: sync send");
    if (d.send_failed) {
        d.send_failed = false;
        throw std::runtime_error(
            "ale_cpp: send failed on the CUDA callback (invalid action?)");
    }

    BatchResult result = d.vectorizer->recv();
    const int64_t batch = static_cast<int64_t>(result.batch_size());
    const int64_t obs_bytes = batch * static_cast<int64_t>(d.vectorizer->stacked_obs_size());

    auto cuda_i32 = at::TensorOptions().dtype(at::kInt).device(at::kCUDA);
    auto cuda_u8 = at::TensorOptions().dtype(at::kByte).device(at::kCUDA);
    auto obs_shape = obs_shape_for(d.vectorizer, batch);
    at::Tensor obs = at::empty(obs_shape, cuda_u8);
    at::Tensor reward = at::empty({batch}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    at::Tensor term = at::empty({batch}, at::TensorOptions().dtype(at::kBool).device(at::kCUDA));
    at::Tensor trunc = at::empty({batch}, at::TensorOptions().dtype(at::kBool).device(at::kCUDA));
    at::Tensor env_id = at::empty({batch}, cuda_i32);
    at::Tensor lives = at::empty({batch}, cuda_i32);
    at::Tensor frame = at::empty({batch}, cuda_i32);
    at::Tensor ep_frame = at::empty({batch}, cuda_i32);

    auto h2d = [&](at::Tensor& dst, const void* src, int64_t bytes, const char* what) {
        cuda_check(p_cudaMemcpyAsync(dst.data_ptr(), src, bytes,
                                     cudaMemcpyHostToDevice, stream),
                   what);
    };
    h2d(obs, result.obs_data(), obs_bytes, "recv: H2D obs");
    h2d(reward, result.rewards_data(), batch * sizeof(float), "recv: H2D reward");
    h2d(term, result.terminations_data(), batch * sizeof(bool), "recv: H2D term");
    h2d(trunc, result.truncations_data(), batch * sizeof(bool), "recv: H2D trunc");
    h2d(env_id, result.env_ids_data(), batch * sizeof(int32_t), "recv: H2D env_id");
    h2d(lives, result.lives_data(), batch * sizeof(int32_t), "recv: H2D lives");
    h2d(frame, result.frame_numbers_data(), batch * sizeof(int32_t), "recv: H2D frame");
    h2d(ep_frame, result.episode_frame_numbers_data(), batch * sizeof(int32_t),
        "recv: H2D ep_frame");

    std::vector<at::Tensor> out = {obs, reward, term, trunc,
                                   env_id, lives, frame, ep_frame};
    if (with_final) {
        at::Tensor final_obs = at::empty(obs_shape, cuda_u8);
        if (result.has_final_obs()) {
            h2d(final_obs, result.final_obs_data(), obs_bytes, "recv: H2D final_obs");
        } else {
            final_obs.zero_();
        }
        out.push_back(final_obs);
    }

    // The H2D copies read from `result`'s host buffers, which are freed when it
    // goes out of scope, so the copies must finish first.
    cuda_check(p_cudaStreamSynchronize(stream), "recv: sync H2D");
    return out;
}

#endif  // BUILD_VECTOR_TORCH_CUDA

// ------------------------------------------------------------------ dispatch
// recv() takes no tensor argument, so the dispatcher cannot pick a backend from
// the inputs. Both ops branch on the handle's target device instead and are
// registered under the device-agnostic CompositeExplicitAutograd key.

at::Tensor ale_send(int64_t handle_id, const at::Tensor& actions) {
    TorchHandle& d = get_handle(handle_id);
#ifdef BUILD_VECTOR_TORCH_CUDA
    if (d.is_cuda) {
        return send_cuda(d, actions);
    }
#endif
    return send_cpu(d, actions);
}

std::vector<at::Tensor> recv_dispatch(TorchHandle& d, bool with_final) {
#ifdef BUILD_VECTOR_TORCH_CUDA
    if (d.is_cuda) {
        return recv_cuda(d, with_final);
    }
#endif
    return recv_cpu(d, with_final);
}

RecvResult ale_recv(int64_t handle_id) {
    auto v = recv_dispatch(get_handle(handle_id), /*with_final=*/false);
    return {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
}

// SameStep autoreset mode: same 8 tensors plus final_obs (the pre-reset
// observation, valid for the envs that are done this step).
RecvResult9 ale_recv_same_step(int64_t handle_id) {
    auto v = recv_dispatch(get_handle(handle_id), /*with_final=*/true);
    return {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]};
}

}  // namespace

TORCH_LIBRARY(ale_cpp, m) {
    m.def("send(int handle_id, Tensor actions) -> Tensor");
    m.def("recv(int handle_id) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("recv_same_step(int handle_id) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(ale_cpp, CompositeExplicitAutograd, m) {
    m.impl("send", &ale_send);
    m.impl("recv", &ale_recv);
    m.impl("recv_same_step", &ale_recv_same_step);
}

void init_vector_module_torch(nb::module_& m) {
    m.def("torch_register_handle",
          [](int64_t handle_id, EnvVectorizer& vectorizer, int64_t pinned_in_ptr,
             int64_t batch_size, bool is_cuda,
             bool continuous, double threshold, int64_t map_action_idx_ptr) {
              // `vectorizer` is the same instance env.ale wraps; nanobind hands
              // us a reference to it, so we just take its address.
              TorchHandle h;
              h.vectorizer = &vectorizer;
              h.pinned_in = reinterpret_cast<void*>(pinned_in_ptr);
              h.batch_size = batch_size;
              h.is_cuda = is_cuda;
              h.continuous = continuous;
              h.threshold = static_cast<float>(threshold);
              h.send_failed = false;
              if (continuous && map_action_idx_ptr) {
                  std::memcpy(h.map_action_idx,
                              reinterpret_cast<const int32_t*>(map_action_idx_ptr),
                              sizeof(h.map_action_idx));
              } else {
                  std::memset(h.map_action_idx, 0, sizeof(h.map_action_idx));
              }
              g_handles[handle_id] = h;
          });
    m.def("torch_unregister_handle",
          [](int64_t handle_id) { g_handles.erase(handle_id); });
#ifdef BUILD_VECTOR_TORCH_CUDA
    m.attr("VECTOR_TORCH_CUDA") = true;
#else
    m.attr("VECTOR_TORCH_CUDA") = false;
#endif
}
