// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4
//
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)

const char *plabel[] = {
"background",
"person",
"bicycle",
"car",
"motorcycle",
"airplane",
"bus",
"train",
"truck",
"boat",
"traffic light",
"fire hydrant",
"N/A",
"stop sign",
"parking meter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"N/A",
"backpack",
"umbrella",
"N/A",
"N/A",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sports ball",
"kite",
"baseball bat",
"baseball glove",
"skateboard",
"surfboard",
"tennis racket",
"bottle",
"N/A",
"wine glass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hot dog",
"pizza",
"donut",
"cake",
"chair",
"couch",
"potted plant",
"bed",
"N/A",
"dining table",
"N/A",
"N/A",
"toilet",
"N/A",
"tv",
"laptop",
"mouse",
"remote",
"keyboard",
"cell phone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"N/A",
"book",
"clock",
"vase",
"scissors",
"teddy bear",
"hair drier",
"toothbrush"
};

class Detector {
 public:
  Detector(const string& model_file,
           const string& weights_file,
           const string& mean_file,
           const string& mean_value);

  std::vector<vector<float> > Detect(const cv::Mat& img);

 private:
  void SetMean(const string& mean_file, const string& mean_value);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
};

Detector::Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
  printf("input_layer: [%d %d]\n", input_geometry_.width, input_geometry_.height);
  /* Load the binaryproto mean file. */
  SetMean(mean_file, mean_value);
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  shared_ptr<Blob<float> > result_blob_tmp;
  Blob<float>* result_blob = net_->output_blobs()[0];
  #if 0
  result_blob_tmp = net_->blob_by_name("layer_19_2_5_mbox_loc/depthwise");
  printf("blob: %s, [%d %d %d]\n", "layer_19_2_5_mbox_loc/depthwise", result_blob_tmp->width(), result_blob_tmp->height(), result_blob_tmp->channels());

  result_blob_tmp = net_->blob_by_name("layer_19_2_5_mbox_conf/depthwise");
  printf("blob: %s, [%d %d %d]\n", "layer_19_2_5_mbox_conf/depthwise", result_blob_tmp->width(), result_blob_tmp->height(), result_blob_tmp->channels());

  result_blob_tmp = net_->blob_by_name("layer_19_2_5_mbox_priorbox");
  printf("blob: %s, [%d %d %d]\n", "layer_19_2_5_mbox_priorbox", result_blob_tmp->width(), result_blob_tmp->height(), result_blob_tmp->channels());  
  #endif
  const float* result = result_blob->cpu_data();
  
  printf("\n[%d %d %d]\n", result_blob->channels(), result_blob->height(), result_blob->width());
  for(int k = 0; k < 7*result_blob->height(); k++)
  {
    if ((0 == k %7) && (0 != k))
        printf("\n");
    printf("%f ", result[k]);
  }
  printf("\n");

  const int num_det = result_blob->height();
  vector<vector<float> > detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    vector<float> detection(result, result + 7);
    detections.push_back(detection);
    result += 7;
  }
  return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Detector::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);


//float *pData = (float *)sample_float.data;
//printf("%f %f %f\n", sample_float.at<cv::Vec3f>(0,0)[0], sample_float.at<cv::Vec3f>(0,0)[1], sample_float.at<cv::Vec3f>(0,0)[2]);
  cv::Mat sample_normalized;
  cv::subtract(sample_float, 127.5, sample_normalized);
  sample_normalized.convertTo(sample_normalized, CV_32F, 1.0 / 127.5, 0.0);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "0,0,0",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
    "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
    "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.35,
    "Only store detections with score higher than the threshold.");

static void writeFileFloat(const char *pFname, float *pData, unsigned size)
{
    FILE* pfile = fopen(pFname, "wb");
    if (!pfile)
    {
        printf("pFileOut open error \n");
        exit(-1);
    }
    for(int i =0; i < size; i++)
    {
        if ((0 != i)&& (0 == (i%16)))
            fprintf(pfile, "\n");
        fprintf(pfile, "%10.6f ", pData[i]);
    }
    fclose(pfile);
}

const char format_head[]=
	"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n\
<annotation>\n\
   <folder>out</folder>\n\
   <filename>NA</filename>\n\
   <path>NA</path>\n\
   <source>\n\
	  <database>Unknown</database>\n\
   </source>\n\
   <size>\n\
	  <width>%d</width>\n\
	  <height>%d</height>\n\
	  <depth>3</depth>\n\
   </size>\n\
   <segmented>0</segmented>\n";

const char format_box[]=
"   <object>\n\
	  <name>alien</name>\n\
	  <pose>Unspecified</pose>\n\
	  <truncated>0</truncated>\n\
	  <difficult>0</difficult>\n\
	  <bndbox>\n\
		 <xmin>%d</xmin>\n\
		 <ymin>%d</ymin>\n\
		 <xmax>%d</xmax>\n\
		 <ymax>%d</ymax>\n\
	  </bndbox>\n\
   </object>\n";

const char format_end[]="</annotation>";

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 0;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Do detection using SSD mode.\n"
        "Usage:\n"
        "    ssd_detect [FLAGS] model_file weights_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/ssd/ssd_detect");
    return 1;
  }

  const string& model_file = argv[1];
  const string& weights_file = argv[2];
  const string& mean_file = FLAGS_mean_file;
  const string& mean_value = FLAGS_mean_value;
  const string& out_file = FLAGS_out_file;
  const float confidence_threshold = FLAGS_confidence_threshold;

  // Initialize the network.
  Detector detector(model_file, weights_file, mean_file, mean_value);

  // Set the output mode.
  std::streambuf* buf = std::cout.rdbuf();
  std::ofstream outfile;
  if (!out_file.empty()) {
    outfile.open(out_file.c_str());
    if (outfile.good()) {
      buf = outfile.rdbuf();
    }
  }
  std::ostream out(buf);
  #if 1
	FILE* pfile = fopen("yiming/dataset/result.txt", "wb");
	if (!pfile)
	{
		printf("pFileOut open error \n");
		exit(-1);
	}
	FILE* pfileSize = fopen("/home/leejohnnie/code/chuanqi305/ssd/yiming/voc_test_1000/size.txt", "wb");
	if (!pfile)
	{
		printf("pFileOut open error \n");
		exit(-1);
	}
	#endif
// Process image one by one.
  std::ifstream infile(argv[3]);
  std::string file;
  while (infile >> file) {
      cv::Mat img = cv::imread(file, -1);
#if 0
      cv::Mat imgResize;
      cv::resize(img, imgResize, cv::Size(300, 300));

	  fprintf(pfileSize, "%d,%d\n", img.cols, img.rows);
	  fflush(pfileSize);

      char buff[256];
      sprintf(buff, "yiming/voc_test_1000%s", strrchr(file.c_str(), '/'));
      imwrite(buff, imgResize);
#endif

#if 1
      //cv::Mat imgSize;
      cv::resize(img, img, cv::Size(300, 300));
      //printf("\n\n\n%d %d %d\n", img.at<cv::Vec3b>(0,0)[0], img.at<cv::Vec3b>(0,0)[1], img.at<cv::Vec3b>(0,0)[2]);
      cv::cvtColor(img, img, CV_BGR2RGB);
      //printf("mean_file: %s, mean_value: %s, confidence_threshold: %f [%d %d]\n", mean_file.c_str(), mean_value.c_str(), confidence_threshold, img.cols, img.rows);

      CHECK(!img.empty()) << "Unable to decode image " << file;
      std::vector<vector<float> > detections = detector.Detect(img);

      /* Print the detection results. */
      //printf("[id, label, score, xmin, ymin, xstd::max, ystd::max]\nDetection cnt: %u\n", (uint32_t)detections.size());
		
			char xmlname[1024];
			char buff[1024];
			strcpy(xmlname, strrchr(file.c_str(), '/')+1);
			*strchr(xmlname, '.') = 0;
			sprintf(buff, "yiming/caffe_result/%s.xml", xmlname);
			printf("xml: %s\n", buff);
			FILE *fpxml = NULL;
			if(NULL == (fpxml = fopen(buff,"ab")))
			{
				printf("open output error!\n");
				return -3;
			}
        fprintf(fpxml, format_head, img.cols, img.rows);
        
      for (int i = 0; i < detections.size(); ++i) {
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xstd::max, ystd::max].
        CHECK_EQ(d.size(), 7);
        const float score = d[2];
        //printf("score: %f\n", score);

        if (score >= confidence_threshold) {

          //fprintf(pfile, "%s %f %f %f %f %f %f\n", strrchr(file.c_str(), '/')+1, d[1], d[2], d[3], d[4], d[5], d[6]);
          //fflush(pfile);
#if 0
          out << "[" << i << "] ";
          out << d[1] << " ";
          out << score << " ";
          out << d[3] << " ";
          out << d[4] << " ";
          out << d[5] << " ";
          out << d[6] << std::endl;
#endif

#if 0
          out << "[" << i << "] ";
          out << static_cast<int>(d[1]) << " ";
          out << score << " ";
          out << static_cast<int>(d[3] * img.cols) << " ";
          out << static_cast<int>(d[4] * img.rows) << " ";
          out << static_cast<int>(d[5] * img.cols) << " ";
          out << static_cast<int>(d[6] * img.rows) << std::endl;
#endif
          int xoffset = 0, yoffset = 0;

#if 0
          xoffset = (static_cast<int>(d[5] * img.cols)- static_cast<int>(d[3] * img.cols)) / 10;
          yoffset = (static_cast<int>(d[6] * img.rows)- static_cast<int>(d[4] * img.rows)) / 10;
#endif

#if 1
            int topx, topy, bottomx,bottomy;
            topx = std::max(static_cast<int>(d[3] * img.cols), 0);
            topx = std::min(topx, img.cols);

            topy = std::max(static_cast<int>(d[4] * img.rows), 0);
            topy = std::min(topy, img.rows);

            bottomx = std::max(static_cast<int>(d[5] * img.cols), 0);
            bottomx = std::min(bottomx, img.cols);

            bottomy = std::max(static_cast<int>(d[6] * img.rows), 0);
            bottomy = std::min(bottomy, img.rows);

          cv::Rect rect = cv::Rect(topx, 
                                   topy, 
                                   bottomx- topx, 
                                   bottomy - topy);
          cv::rectangle(img, rect, cv::Scalar(0, 255, 0));
          
			char text[256];
			sprintf(text, "%d, %s %.3f", static_cast<int>(d[1]), plabel[static_cast<int>(d[1])], score);

			int baseLine = 0;
			cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

			int x = static_cast<int>(d[3] * img.cols) - xoffset;
			int y = static_cast<int>(d[4] * img.rows) - yoffset - label_size.height - baseLine;
			if (y < 0)
				y = 0;
			if (x + label_size.width > img.cols)
				x = img.cols - label_size.width;

			cv::rectangle(img, cv::Rect(cv::Point(x, y),
						  cv::Size(label_size.width, label_size.height + baseLine)),
						  cv::Scalar(255, 255, 255), CV_FILLED);

			cv::putText(img, text, cv::Point(x, y + label_size.height),
						cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
#endif
{
            int topx, topy, bottomx,bottomy;
            topx = std::max(static_cast<int>(d[3] * img.cols), 0);
            topx = std::min(topx, img.cols);

            topy = std::max(static_cast<int>(d[4] * img.rows), 0);
            topy = std::min(topy, img.rows);

            bottomx = std::max(static_cast<int>(d[5] * img.cols), 0);
            bottomx = std::min(bottomx, img.cols);

            bottomy = std::max(static_cast<int>(d[6] * img.rows), 0);
            bottomy = std::min(bottomy, img.rows);

			fprintf(fpxml, format_box, 
			       topx,
			       topy,
			       bottomx,
			       bottomy
			       );
}

        }

        //getchar();
      }
			fprintf(fpxml, format_end);
		    fclose(fpxml);
#endif
      imshow("d", img);
      cv::waitKey();
      
      //imwrite("/home/leejohnnie/code/ssd/caffe/examples/images/ssd.jpg", img);
  }
  fclose(pfile);
  fclose(pfileSize);
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
