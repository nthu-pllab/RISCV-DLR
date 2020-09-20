/*!
 *  Copyright (c) 2015 by Contributors
 * \file thread_local.h
 * \brief Portable thread local storage.
 */
#ifndef DMLC_THREAD_LOCAL_H_
#define DMLC_THREAD_LOCAL_H_

#include "./base.h"
#include <memory>
#include <mutex>
#include <vector>

namespace dmlc {

// macro hanlding for threadlocal variables
#ifdef __GNUC__
#define MX_THREAD_LOCAL __thread
#elif __STDC_VERSION__ >= 201112L
#define MX_THREAD_LOCAL _Thread_local
#elif defined(_MSC_VER)
#define MX_THREAD_LOCAL __declspec(thread)
#endif

#if DMLC_CXX11_THREAD_LOCAL == 0
#pragma message("Warning: CXX11 thread_local is not formally supported")
#endif

/*!
 * \brief A threadlocal store to store threadlocal variables.
 *  Will return a thread local singleton of type T
 * \tparam T the type we like to store
 */
template <typename T>
class ThreadLocalStore {
public:
  /*! \return get a thread local singleton */
  static T *Get() {
#if DMLC_CXX11_THREAD_LOCAL
    static thread_local T inst;
    return &inst;
#else
    static MX_THREAD_LOCAL T *ptr = nullptr;
    if (ptr == nullptr) {
      ptr = new T();
      Singleton()->RegisterDelete(ptr);
    }
    return ptr;
#endif
  }

private:
  /*! \brief constructor */
  ThreadLocalStore() {}
  /*! \brief destructor */
  ~ThreadLocalStore() {
    for (size_t i = 0; i < data_.size(); ++i) {
      delete data_[i];
    }
  }
  /*! \return singleton of the store */
  static ThreadLocalStore<T> *Singleton() {
    static ThreadLocalStore<T> inst;
    return &inst;
  }
  /*!
   * \brief register str for internal deletion
   * \param str the string pointer
   */
  void RegisterDelete(T *str) {
    /* [RISCV-DLR] multi-threading is not supported, so lock is unnecessary */
    //std::unique_lock<std::mutex> lock(mutex_);
    data_.push_back(str);
    //lock.unlock();
  }

  /*! \brief internal mutex */
  /* [RISCV-DLR] multi-threading is not supported, so lock is unnecessary */
  //std::mutex mutex_;

  /*!\brief internal data */
  std::vector<T *> data_;
};

} // namespace dmlc

#endif // DMLC_THREAD_LOCAL_H_
