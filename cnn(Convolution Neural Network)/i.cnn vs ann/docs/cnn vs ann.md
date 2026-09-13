| Feature               | ANN                                       | CNN                                                            |
| --------------------- | ----------------------------------------- | -------------------------------------------------------------- |
| Full name             | Artificial Neural Network                 | Convolutional Neural Network                                   |
| Main idea             | Learn relationships between inputs        | Learn spatial features and patterns                            |
| Common input          | Tabular/vector data                       | Images, video, spatial data                                    |
| Main layer            | Fully Connected/Dense                     | Convolution + Pooling + Dense                                  |
| Spatial information   | Usually not preserved                     | Preserved and exploited                                        |
| Feature extraction    | Usually learned through dense connections | Automatically learned using convolution                        |
| Parameters for images | Can become very large                     | Usually much fewer for images                                  |
| Translation handling  | Weak                                      | Better due to convolution/pooling and learned spatial patterns |
| Weight sharing        | ❌ No                                      | ✅ Yes, convolution filters share weights                       |
| Local connectivity    | ❌ Usually no                              | ✅ Yes                                                          |
| Typical use           | Classification, regression, tabular data  | Image classification, object detection, segmentation           |

# Total pixels=Height×Width