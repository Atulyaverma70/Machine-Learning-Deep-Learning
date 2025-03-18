

# General Questions (Both TensorFlow & PyTorch)

2. When would you choose TensorFlow over PyTorch and vice versa?
* Use TensorFlow for production, mobile, and large-scale distributed training.
* Use PyTorch for rapid prototyping, debugging, and research.
3. What are tensors, and how do they differ from NumPy arrays?
* Tensors are multi-dimensional arrays used in deep learning.
* Unlike NumPy arrays, tensors support GPU acceleration and automatic differentiation.
4. Explain Autograd (PyTorch) and AutoDiff (TensorFlow).
* Autograd (PyTorch): Automatically computes gradients for backpropagation using dynamic computation graphs.
* AutoDiff (TensorFlow): Uses tf.GradientTape to record operations and compute gradients.
5. What is a computational graph, and why is it important?
* A computational graph represents mathematical operations as a directed graph.
* It helps in automatic differentiation, optimizing performance, and parallel execution.
6. Explain eager execution vs. static computation graphs.
* Eager execution (default in PyTorch) runs operations immediately.
* Static graphs (TensorFlow) define a graph first and execute it later, improving performance.
7. How do you optimize the training of deep learning models?
* Use gradient clipping, batch normalization, learning rate scheduling, and data augmentation.
Apply mixed-precision training for performance.
Optimize model architecture using dropout, residual connections, etc.
TensorFlow-Specific Questions
8. What are tf.data, tf.function, and tf.GradientTape?
* tf.data: Efficient data loading and preprocessing pipeline.
* tf.function: Converts Python functions into optimized TensorFlow graphs.
* tf.GradientTape: Computes gradients dynamically during training.
9. What is TensorFlow Lite (TFLite), and when would you use it?
* A lightweight version of TensorFlow optimized for mobile and edge devices.
Used in Android, iOS, Raspberry Pi, and embedded systems.
10. Explain the use of tf.keras in TensorFlow.
* tf.keras is TensorFlow’s high-level API for building deep learning models.
* It simplifies model creation, training, and evaluation.
11. How do you distribute training across multiple GPUs in TensorFlow?
python
Copy
Edit
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = tf.keras.models.Sequential([...])  # Define model inside scope
    model.compile(optimizer='adam', loss='categorical_crossentropy')
12. What is a SavedModel format, and how do you load a trained model?
SavedModel is TensorFlow’s standard format for saving models.
python
Copy
Edit
model.save("saved_model")
loaded_model = tf.keras.models.load_model("saved_model")
13. What are tf.placeholder and tf.variable?
tf.placeholder (used in TensorFlow 1.x) is a placeholder for input data.
tf.Variable stores trainable parameters (weights/biases).
14. How does TensorFlow handle ONNX models?
ONNX (Open Neural Network Exchange) allows exporting models between TensorFlow and PyTorch.
bash
Copy
Edit
pip install tf2onnx
python
Copy
Edit
import tf2onnx
onnx_model = tf2onnx.convert.from_keras(model)
15. Explain how TensorFlow supports mobile and edge computing.
TensorFlow supports TFLite and TensorFlow.js for mobile and web applications.
PyTorch-Specific Questions
16. How does PyTorch handle dynamic computation graphs?
PyTorch uses eager execution, dynamically adjusting the computation graph during runtime.
17. What is the role of torch.nn, torch.optim, and torch.autograd?
torch.nn: Defines neural network layers.
torch.optim: Provides optimization algorithms (Adam, SGD).
torch.autograd: Computes gradients automatically.
18. How do you perform model serialization (torch.save, torch.load)?
python
Copy
Edit
torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth"))
19. What is a DataLoader in PyTorch?
A DataLoader provides efficient data batching and shuffling.
python
Copy
Edit
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
20. How does torch.no_grad() improve inference performance?
Disables gradient calculation, reducing memory usage and speeding up inference.
python
Copy
Edit
with torch.no_grad():
    predictions = model(input_data)
21. What is the difference between model.eval() and model.train()?
model.train(): Enables dropout and batch normalization during training.
model.eval(): Disables them for inference.
22. How can you perform distributed training in PyTorch?
Use torch.nn.parallel.DistributedDataParallel for multi-GPU training.
23. Explain torchvision and torchaudio.
torchvision: Pretrained models and image processing utilities.
torchaudio: Audio processing and speech recognition utilities.
Advanced Topics (Both TensorFlow & PyTorch)
24. How do you fine-tune a pre-trained model in TensorFlow/PyTorch?
Load a pretrained model, freeze some layers, and retrain the last few layers.
25. What are GANs, and how would you implement them in TensorFlow/PyTorch?
Generative Adversarial Networks (GANs) consist of a generator and discriminator competing against each other.
Implemented using torch.nn (PyTorch) or tf.keras (TensorFlow).
26. Explain Transformer models (BERT, GPT) and their implementation.
Transformers use self-attention mechanisms for NLP tasks.
Implemented using transformers library (Hugging Face).
27. How do you optimize deep learning models for deployment?
Quantization (reducing precision), pruning (removing unnecessary weights), and model distillation.
28. Explain quantization and pruning in deep learning models.
Quantization reduces model size by converting weights from float32 to int8.
Pruning removes redundant neurons to improve efficiency.
29. What are the differences between FP32, FP16, and INT8 in model performance?
FP32 (Full Precision): High accuracy but slow.
FP16 (Half Precision): Faster with slight accuracy drop.
INT8 (Quantized): Best for edge devices, fastest but lowest precision.
