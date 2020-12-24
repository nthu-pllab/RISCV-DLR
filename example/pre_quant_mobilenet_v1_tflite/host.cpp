#include <dlr.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "kernel.inc"

// #[TODO]
#define DLR_DIR "/path/to/DLR/repo"

using namespace std;

const int img_size = 150528;
const int label_range = 1001;

int test_start = 0;      // dataset's start
int test_runs = 10;      // run for how many times
int trial = 0;           // number of trial
int successful_runs = 0; // number of successful run

double test_success_count = 0.0;
uint8_t input_Placeholder[img_size];
uint8_t *outputs;

ifstream label_f(DLR_DIR "/dataset/val.txt");

string get_file_name(int n, string prefix) {
  int sequence[8];
  int number = n;
  for (int j = 0; j < 8; j++) {
    int digit = number % 10;
    sequence[j] = digit;
    number = (number - digit) / 10;
  }

  string file_name = prefix;
  for (int j = 7; j >= 0; j--) {
    file_name += to_string(sequence[j]);
  }

  return file_name;
}

bool read_imgs(string label, uint8_t *input) {
  string fileName = label + ".bin";
  cout << "filename : " << fileName << "\n";
  FILE *image_file;
  image_file = fopen(fileName.c_str(), "rb");

  if (image_file == NULL)
    return false;

  memset(input, 0, img_size);
  fread(input, 1, img_size, image_file);

  return true;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Please enter file\n");
    return -1;
  }
  string json = argv[1];
  string params = argv[2];
  string lib = argv[3];
  dlr::DLR dlr;
  dlr.Build(json, params, lib, DLRBackend::kBAREMETAL);
  dlr.InitOp();

  FILE *image_test;
  FILE *image_test_label;

  unsigned char label_buf[1];

  bool load_success = false;

  // load label if test_start != 0
  if (test_start != 0 && label_f.is_open()) {
    int count = 0;
    while (count != test_start) {
      string trash;
      getline(label_f, trash);
      ++count;
    }
  }

  for (int it = test_start; (successful_runs < test_runs) && (trial < test_runs + 100); it++, trial++) {

    string file_name = get_file_name(it + 1, "tflite_quant_ILSVRC2012_val_");

    string label_s;
    if (label_f.is_open()) {
      getline(label_f, label_s);
    }

    int ground_true;
    if (label_s.find("JPEG") != string::npos) {
      ground_true = stoi(label_s.substr(label_s.find("JPEG") + 5));
    } else {
      cerr << "load label failed !\n";
      exit(0);
    }

    cout << "Read : " << file_name << endl;
    load_success = read_imgs(DLR_DIR "/dataset/imagenet_quant_224_tflite/" + file_name, input_Placeholder);

    // if load failed
    if (!load_success) {
      cout << "load fail\n";
      continue;
    }

    DLDataType dtype = {kDLUInt, 8, 1};
    vector<int> input_shape = {1, 224, 224, 3};
    dlr.SetInputPtr(0, (char *)input_Placeholder, input_shape, &dtype);

    dlr.Run();

    outputs = (uint8_t *)dlr.GetOutputPtr(0);

    // get predict result
    uint8_t max_value = 0;
    int max_index = 0;
    for (int k = 0; k < label_range; k++) {
      if (outputs[k] > max_value) {
        max_value = outputs[k];
        max_index = k;
      }
    }

    --max_index;
    cout << "predict ANW: " << max_index << "\n";
    cout << "correct ANW: " << ground_true << "\n";

    if (max_index == ground_true) {
      cout << "Correct\n";
      test_success_count++;
    } else {
      cout << "Fail\n";
    }

    successful_runs++;
  }
  //---------------------------------

  cout << endl;
  cout << "successful_runs : " << successful_runs << "\n";
  cout << "The success rate: " << test_success_count / successful_runs << "\n";

  return 0;
}
