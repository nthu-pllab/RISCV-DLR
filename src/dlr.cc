/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "dlr.h"

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/serializer.h>

#include <algorithm>
#include <cfloat>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace dlr {
void DLR::SetInputPtr(int op, int p_num, void *data_in, DLDataType *setType) {
  assert(p_num < tvm_graph_runtime_->op_exe[op].args.size());
  DLTensor *input;
  int dtype_code = setType->code;
  int dtype_bits = setType->bits;
  int dtype_lanes = setType->lanes;
  int device_type = kDLCPU;
  int device_id = 0;
  int in_ndim = tvm_graph_runtime_->op_exe[op].args[p_num].ndim;
  size_t curr_size = tvm::runtime::GetDataSize(tvm_graph_runtime_->op_exe[op].args[p_num]);
  TVMArrayAlloc(tvm_graph_runtime_->op_exe[op].args[p_num].shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
  TVMArrayCopyFromBytes(input, data_in, curr_size);
  void *target_ptr = tvm_graph_runtime_->op_exe[op].args[p_num].data;
  assert(opa != 0);
  assert(target_ptr != 0);

  //updata tensor
  int find_num = 0;
  for (uint32_t i = 0; i < tvm_graph_runtime_->op_exe.size(); i++) {
    for (uint32_t j = 0; j < tvm_graph_runtime_->op_exe[i].args.size(); j++) {
      if (tvm_graph_runtime_->op_exe[i].args[j].data == target_ptr) {
        tvm_graph_runtime_->op_exe[i].args[j] = *input;
        TVMValue v;
        DLTensor *t = &tvm_graph_runtime_->op_exe[i].args[j];
        v.v_handle = t;
        tvm_graph_runtime_->op_exe[op].arg_values[j] = v;
        find_num++;
      }
    }
  }
  //std::cout << "find num of target : " << find_num << std::endl;
}

void DLR::GetOutputPtr(int op, int p_num, void **data_out, DLDataType **getType) {
  assert(p_num < tvm_graph_runtime_->op_exe[op].args.size());
  *getType = &tvm_graph_runtime_->op_exe[op].args[p_num].dtype;
  *data_out = tvm_graph_runtime_->op_exe[op].args[p_num].data;
}

void DLR::SetInputPtr(int index, char *data_in, std::vector<int> &shape) {
  DLTensor *input;
  constexpr int dtype_code = kDLFloat;
  constexpr int dtype_bits = 32;
  constexpr int dtype_lanes = 1;
  constexpr int device_type = kDLCPU;
  constexpr int device_id = 0;
  int in_ndim = shape.size();
  int64_t *in_shape = new int64_t[in_ndim];
  int num = 1;

  for (int i = 0; i < in_ndim; i++) {
    in_shape[i] = shape[i];
    num = num * shape[i];
  }

  TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
  TVMArrayCopyFromBytes(input, data_in, num * sizeof(float));
  SetInput(index, input);
  delete[] in_shape;
}

void DLR::SetInputPtr(int index, void *data_in, std::vector<int> &shape, DLDataType *setType) {
  DLTensor *input;
  int dtype_code = setType->code;
  int dtype_bits = setType->bits;
  int dtype_lanes = setType->lanes;
  int device_type = kDLCPU;
  int device_id = 0;
  int in_ndim = shape.size();
  int64_t *in_shape = new int64_t[in_ndim];
  int num = 1;

  for (int i = 0; i < in_ndim; i++) {
    in_shape[i] = shape[i];
    num = num * shape[i];
  }

  TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &input);
  TVMArrayCopyFromBytes(input, data_in, num * dtype_bits / 8);
  SetInput(index, input);
  delete[] in_shape;
}

void *DLR::GetOutputPtr(int index) {
  DLTensor *p_DLTensor = (DLTensor *)(tvm_graph_runtime_->GetOutput(index).operator->());
  return (*p_DLTensor).data;
}

void *DLR::GetOutputPtr(tvm::runtime::GraphRuntime::OpArgs &op, int p_num) {
  assert(p_num < op.args.size());
  return op.args[p_num].data;
}

void DLR::SetInput(int index, DLTensor *data_in) {
  tvm_graph_runtime_->SetInput(index, data_in);
}

int64_t DLR::GetShpaeSize(const std::string &name) {
  int in_idx = tvm_graph_runtime_->GetInputIndex(name);
  size_t shape_size = 1;

  for (int64_t sz : tvm_graph_runtime_->attrs_.shape[in_idx])
    shape_size *= static_cast<size_t>(sz);

  return shape_size;
}

int64_t DLR::GetShpaeSize(tvm::runtime::GraphRuntime::OpArgs &op, int p_num) {
  assert(p_num < op.args.size());
  size_t shape_size = 1;

  for (int64_t i = 0; i < op.args[p_num].ndim; i++)
    shape_size *= op.args[p_num].shape[i];

  return shape_size;
}

std::vector<int64_t> DLR::GetShpae(const std::string &name) {
  int in_idx = tvm_graph_runtime_->GetInputIndex(name);
  std::vector<int64_t> temp;

  for (int64_t sz : tvm_graph_runtime_->attrs_.shape[in_idx]) {
    temp.push_back(sz);
  }

  return temp;
}

void DLR::InitOp() {
  int index = 0;
  opa = new op[tvm_graph_runtime_->GetNumOfNodes()];
  for (uint32_t nid = 0; nid < tvm_graph_runtime_->GetNumOfNodes(); ++nid) {
    const auto &inode = tvm_graph_runtime_->nodes_[nid];
    if (inode.op_type == "null")
      continue;
    opa[index].values = tvm_graph_runtime_->op_exe[index].arg_values.data();
    opa[index].tcodes = tvm_graph_runtime_->op_exe[index].arg_tcodes.data();
    opa[index].num = tvm_graph_runtime_->op_exe[index].arg_values.size();
    index++;
  }
}

void DLR::Init(const std::string &graph_json,
               tvm::runtime::Module module,
               const std::vector<TVMContext> &ctxs) {
  tvm_graph_runtime_ = std::make_shared<tvm::runtime::GraphRuntime>();
  tvm_graph_runtime_->Init(graph_json, module, ctxs);
}

void DLR::LoadParams(const std::string &param_blob) {
  std::ifstream par(param_blob.c_str(), std::ios::binary);
  std::stringstream ss;
  ss << par.rdbuf();
  std::string params_data = ss.str();
  par.close();
  dmlc::MemoryStringStream strm(const_cast<std::string *>(&params_data));
  tvm_graph_runtime_->LoadParams(&strm);
}

DLTensor DLR::GetInput(const std::string &name) {
  int index = tvm_graph_runtime_->GetInputIndex(name);
  tvm::runtime::NDArray p = tvm_graph_runtime_->GetInput(index);
  DLTensor p_DLTensor = *(p.operator->());
  return p_DLTensor;
}

void DLR::dump_kernel() {
  backend_handle = new BareMetal;
  BareMetal *BM = (BareMetal *)backend_handle;
  int index = 0;

  BM->main_section.append("void dlr::DLR::Run()\n");
  BM->main_section.append("{\n");
  BM->main_section.append("    int32_t ret;\n");
  for (uint32_t nid = 0; nid < tvm_graph_runtime_->GetNumOfNodes(); ++nid) {
    const auto &inode = tvm_graph_runtime_->nodes_[nid];
    if (inode.op_type == "null")
      continue;
    assert(inode.op_type == "tvm_op");

    std::string extern_ins;
    extern_ins.append("extern \"C\" int32_t " + inode.param.func_name);
    extern_ins.append("(");

    // output in here
    extern_ins.append("void* args, void* arg_type_ids, int32_t num_args");
    extern_ins.append(");\n");
    BM->extern_section.append(extern_ins);

    // main
    std::string op_ins;
    op_ins.append("    ret = " + inode.param.func_name);
    op_ins.append("(");
    // output in here
    std::string num = std::to_string(index);
    op_ins.append("opa[" + num + "].values");
    op_ins.append(", opa[" + num + "].tcodes");
    op_ins.append(", opa[" + num + "].num");

    op_ins.append(");\n");
    op_ins.append("    assert(ret == 0);\n");
    BM->main_section.append(op_ins);
    index++;
  }
  BM->main_section.append("}\n");

  BM->kernel = BM->include_section + BM->extern_section + BM->init_section +
               "\n" + BM->main_section;
  std::cout << BM->kernel << std::endl;
}

void DLR::Build(const std::string &json_path, const std::string &params, const std::string &lib, const DLRBackend backend) {
  std::ifstream ifs(json_path);
  std::string json((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
  tvm::runtime::Module m;
  std::vector<TVMContext> ctxs;
  TVMContext cpu_ctx;
  cpu_ctx.device_type = kDLCPU;
  cpu_ctx.device_id = 0;
  ctxs.push_back(cpu_ctx);

  // handle params
  Init(json, m, ctxs);
  LoadParams(params);
}
} // namespace dlr
