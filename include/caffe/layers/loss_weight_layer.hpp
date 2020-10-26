#ifndef CAFFE_LOSS_WEIGHT_LAYER_HPP_
#define CAFFE_LOSS_WEIGHT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class LossWeightLayer : public Layer<Dtype> {
 public:
  explicit LossWeightLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "LossWeight"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

 };
} // namespace caffe


#endif  // CAFFE_LOSS_WEIGHT_LAYER_HPP_
