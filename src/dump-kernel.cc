#include <assert.h>
#include <iostream>
#include <string>

#include <dlr.h>

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cout << "Usage: " << argv[0] << "<model-graph> <model-param> <model-lib>\n";
    return -1;
  }
  std::string json = argv[1];
  std::string params = argv[2];
  std::string lib = argv[3];

  dlr::DLR dlr;
  dlr.Build(json, params, lib, DLRBackend::kBAREMETAL);
  dlr.dump_kernel();

  return 0;
}
