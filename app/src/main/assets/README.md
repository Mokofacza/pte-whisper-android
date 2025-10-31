---
license: apache-2.0
---

# Introduction

This repository hosts the [whisper-small](https://huggingface.co/openai/whisper-small) model for the [React Native ExecuTorch](https://www.npmjs.com/package/react-native-executorch) library. It includes the model exported for xnnpack backend in `.pte` format, ready for use in the **ExecuTorch** runtime.

If you'd like to run these models in your own ExecuTorch runtime, refer to the [official documentation](https://pytorch.org/executorch/stable/index.html) for setup instructions.

## Compatibility

If you intend to use this models outside of React Native ExecuTorch, make sure your runtime is compatible with the **ExecuTorch** version used to export the `.pte` files. For more details, see the compatibility note in the [ExecuTorch GitHub repository](https://github.com/pytorch/executorch/blob/11d1742fdeddcf05bc30a6cfac321d2a2e3b6768/runtime/COMPATIBILITY.md?plain=1#L4). If you work with React Native ExecuTorch, the constants from the library will guarantee compatibility with runtime used behind the scenes.

These models were exported using v0.6.0 version of ExecuTorch and **no forward compatibility** is guaranteed. Older versions of the runtime may not work with these files.