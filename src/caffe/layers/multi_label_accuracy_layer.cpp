#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/multilabel_sigmoid_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "The data and label should have the same number of instances";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())
    << "The data and label should have the same number of channels";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
    << "The data and label should have the same height";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
    << "The data and label should have the same width";
  // Top will contain:
  // top[0] = Recall (TP/TP+FN),
  // top[1] = Precision (TP / (TP + FP))
  // top[2] = F1 Score (2 TP / (2 TP + FP + FN))
   top[0]->Reshape(1, 3, 1, 1);
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  /*Dtype true_positive = 0;
  Dtype false_positive = 0;
  Dtype true_negative = 0;
  Dtype false_negative = 0;*/
  int true_positive = 0;
  int false_positive = 0;
  int true_negative = 0;
  int false_negative = 0;
  int count_pos = 0;
  int count_neg = 0;
  //Dtype accuracy = 0.0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  // Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int count = bottom[0]->count();

  for (int ind = 0; ind < count; ++ind) {
    // Accuracy
    int label = static_cast<int>(bottom_label[ind]+0.5); //added 0.5 to immitate round of behavior
    int est = static_cast<int>(bottom_data[ind]+0.7);
    //if (label == est) {
    //  ++accuracy;
    //}
   //}
    if (label > 0) {
    // Update Positive accuracy and count
      true_positive += (est > 0);
      false_negative += (est <= 0);
      count_pos++;
    }
    if (label <= 0) {
    // Update Negative accuracy and count
      true_negative += (est <= 0);
      false_positive += (est > 0);
      count_neg++;
    }
  }
  //Dtype sensitivity = (count_pos > 0)? (true_positive / count_pos) : 0;
  //Dtype specificity = (count_neg > 0)? (true_negative / count_neg) : 0;
  //Dtype harmmean = ((count_pos + count_neg) > 0)?
  //  2 / (count_pos / true_positive + count_neg / true_negative) : 0;
  Dtype recall = (true_positive > 0)?
    (float(true_positive) / (true_positive + false_negative)) : 0;
  Dtype precision = (true_positive > 0)?
    (float(true_positive) / (true_positive + false_positive)) : 0;
  Dtype f1_score = (true_positive > 0)?
    2 * float(true_positive) /
    (2 * true_positive + false_positive + false_negative) : 0;

  //DLOG(INFO) << "Sensitivity: " << sensitivity;
  //DLOG(INFO) << "Specificity: " << specificity;
  DLOG(INFO) << "Recall: " << recall;
  DLOG(INFO) << "Precision: " << precision;
  DLOG(INFO) << "F1 Score: " << f1_score;

  //top[0]->mutable_cpu_data()[0] = sensitivity;
  //top[0]->mutable_cpu_data()[1] = true_negative;
  //top[0]->mutable_cpu_data()[2] = true_positive;
  top[0]->mutable_cpu_data()[0] = recall; 
  top[0]->mutable_cpu_data()[1] = precision;
  top[0]->mutable_cpu_data()[2] = f1_score;

  //DLOG(INFO) << "Accuracy: " << accuracy/count;
  //top[0]->mutable_cpu_data()[0] = accuracy / count;
  //by hueifang. July 1, 2015
  //(*top)[0]->mutable_cpu_data()[5] = true_positive;
  //(*top)[0]->mutable_cpu_data()[6] = true_negative;
  //(*top)[0]->mutable_cpu_data()[7] = count_pos;
  //(*top)[0]->mutable_cpu_data()[8] = count_neg;
  // MultiLabelAccuracy should not be used as a loss function.
  //return Dtype(0);
}


INSTANTIATE_CLASS(MultiLabelAccuracyLayer);
REGISTER_LAYER_CLASS(MultiLabelAccuracy);

}  // namespace caffe

