#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <torch/script.h>
#include <dlfcn.h>
int main(int argc, char** argv) {

	//if (argc != 4) {
	//	std::cout << argv[0] << " image.jpg end_to_end_model.pt libmaskrcnn_benchmark_customops.so" << std::endl;
	//	return 1;
	//}
	
	//void* custom_op_lib = dlopen(argv[3], RTLD_NOW | RTLD_GLOBAL);
	
	//void* custom_op_lib = dlopen("custom_ops.cp36-win_amd64.dll", RTLD_NOW | RTLD_GLOBAL);
	void* custom_op_lib = dlopen("custom_ops_no_opencv.dll", RTLD_NOW | RTLD_GLOBAL);
	if (custom_op_lib == NULL) {
		std::cerr << "could not open custom op library: " <<dlerror() <<std::endl;
		return 1;
	}
	
	auto img_ = cv::imread(argv[1], cv::IMREAD_COLOR);
	cv::Mat img(800, 800, CV_8UC3);
	cv::resize(img_, img, img.size(), 0, 0, cv::INTER_AREA);
	auto input_ = torch::tensor(at::ArrayRef<uint8_t>(img.data, img.rows * img.cols * 3)).view({ img.rows, img.cols, 3 });
	std::shared_ptr<torch::jit::script::Module> module =
		torch::jit::load("D:\\github\\weights\\mask_struct_model.pt");

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(input_);
	auto elements = module->forward(inputs).toTuple()->elements();
	//std::cout << elements << std::endl;
	enum {
		Box = 0,
		Label = 1,
		Mask = 2,
		Score = 3,
	};

	auto tensor_masks = elements[Mask].toTensor();
	auto tensor_boxes = elements[Box].toTensor();
	auto tensor_labels = elements[Label].toTensor().toType(at::ScalarType::Int); //long编译拿不到数据，编译不了
	int mask_shape = tensor_masks.size(2);
	int object_num = tensor_masks.size(0);
	int padding = 1;
	float threashold = 0.5f;
	cv::Mat boxes(object_num, 4, CV_32F, tensor_boxes.data<float>());
	auto labels_ptr = tensor_labels.accessor<int, 1>();
	float scale = 1 + 2 * padding / (float)(2 * padding + mask_shape);
	float bbox_scale_w = float(img_.cols) / img.cols;
	float bbox_scale_h = float(img_.rows) / img.rows;
	cv::Mat mask;
	cv::Mat kernal = cv::Mat::ones(3, 3, CV_32F) * (1.0f/9.0f);	
	cv::Mat out_mask_img;
	out_mask_img.create(img_.size(), CV_8UC2);
	out_mask_img = cv::Scalar::all(0);
	for (int c = 0; c < object_num; ++c) {		
		mask.create(mask_shape + 2 * padding, mask_shape + 2 * padding,CV_32F);
		mask = cv::Scalar::all(0);
		for (int row = padding; row < mask.rows-padding; ++row) {
			for (int col = padding; col < mask.cols- padding; ++col) {
				mask.at<float>(row, col) = tensor_masks.accessor<float, 4>()[c][0][row- padding][col- padding];
			}
		}
		auto& bbox = std::vector<float>(boxes.row(c));
		auto center_x = (bbox[2] + bbox[0]) * 0.5f;
		auto center_y = (bbox[3] + bbox[1]) * 0.5f;
		auto w_2 = (bbox[2] - bbox[0]) * 0.5f * scale;
		auto h_2 = (bbox[3] - bbox[1]) * 0.5f * scale;
		auto bbox_scaled = std::vector<float>{ (center_x - w_2)*bbox_scale_w, (center_y - h_2)*bbox_scale_h,
			(center_x + w_2)*bbox_scale_w, (center_y + h_2)*bbox_scale_h };
		int TO_REMOVE = 1;
		auto w = std::max(int(bbox_scaled[2] - bbox_scaled[0] + TO_REMOVE), 1);
		auto h = std::max(int(bbox_scaled[3] - bbox_scaled[1] + TO_REMOVE), 1);
		cv::resize(mask, mask, cv::Size(w, h));
		cv::filter2D(mask, mask, CV_32F, kernal);
		cv::threshold(mask, mask, threashold, 255, 0);
		mask.convertTo(mask, CV_8U);
		/*cv::Mat erode_mask;
		cv::dilate(mask, erode_mask,cv::getStructuringElement(0,cv::Size(3,3)));
		cv::absdiff(mask, erode_mask, mask);*/
		for (int row = 0; row < h; ++row) {
			for (int col = 0; col < w; ++col) {
				int x = col+ bbox_scaled[0];
				int y = row + bbox_scaled[1];
				if (x < 0 || x>=  img_.cols || y < 0 || y>=img_.rows) continue;
				if (mask.at<uchar>(row, col) == 255) {
					out_mask_img.ptr<uchar>(y, x)[labels_ptr[c]-1] = 255;
				}
			}
		}
	}
	
	//cv::imshow("mask", out_mask_img);
	//img_.convertTo(img_, CV_8UC4);
	cv::Mat erode_mask_img;
	cv::erode(out_mask_img, erode_mask_img, cv::getStructuringElement(0,cv::Size(5,5)));
	cv::absdiff(erode_mask_img, out_mask_img,out_mask_img);
	for (int row = 0; row < img_.rows; ++row) {
		for (int col = 0; col < img_.cols; ++col) {
			if (out_mask_img.ptr<uchar>(row, col)[0] == 255) img_.at<cv::Point3_<uchar>>(row, col) = { 0,0,255 };
			if (out_mask_img.ptr<uchar>(row, col)[1] == 255) img_.at<cv::Point3_<uchar>>(row, col) = { 255,0,255 };
		}
	}
	imshow("铁路-maskrcnn-windows-cuda-C++测试", img_);
	cv::waitKey();
		
	//auto data_accessor = tensor_masks.accessor<float, 4>();
	//for (int c = 0; c < channels; ++c) {
	//	for (int row = 0; row < rows; ++row) {
	//		for (int col = 0; col < cols; ++col) {
	//			masks.at<float>(row, col, c) = data_accessor[c][0][row][col];
	//		}
	//	}
	//}
	
	//auto res = module->forward(inputs).toTensor();

	//cv::Mat cv_res(res.size(0), res.size(1), CV_8UC3, (void*)res.data<uint8_t>());
	//cv::namedWindow("Detected", cv::WINDOW_AUTOSIZE);
	//cv::imshow("Detected", cv_res);

	//cv::waitKey(0);
	return 0;
}
