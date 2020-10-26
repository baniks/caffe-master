#include <vector>

#include "caffe/layers/loss_weight_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void LossWeightLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
top[0]->Reshape(bottom[0]->shape());
}


template <typename Dtype>
void LossWeightLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void LossWeightLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
 const Dtype* bottom_label = bottom[0]->cpu_data();
 const int count = bottom[0]->count(); //total number of elements
 const int num = bottom[0]->num();     //number of batches
 int dim = count/num;
 Dtype* top_data = top[0]->mutable_cpu_data();
 vector<Dtype> weight_vec(dim);

 for (int j = 0; j < dim; ++j) {
    for (int i = 0; i < num; ++i) {
        if (bottom_label[ i*dim + j ] > 0) {
            weight_vec[j]+=1;            
        }
    } 
 }
 Dtype scale_factor = -(1.0/num);
 Dtype* weight=&weight_vec[0];
 caffe_scal(dim, scale_factor, weight);
 caffe_add_scalar(dim, Dtype(1), weight);
 vector<Dtype> final_weight;
 for (int i = 0; i < num; ++i){
    final_weight.insert(final_weight.end(),weight_vec.begin(),weight_vec.end());
 }
/* vector<Dtype> test(count);
 for (int i=0; i<count;i++){
    test[i]=i;
 }
 Dtype* test_ptr=&test[0];*/
 caffe_copy(count, &final_weight[0], top_data);
 // caffe_copy(count, test_ptr, top_data);
}


INSTANTIATE_CLASS(LossWeightLayer);
REGISTER_LAYER_CLASS(LossWeight);

}  // namespace caffe
 
