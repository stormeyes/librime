//
// Copyright RIME Developers
// Distributed under the BSD License
//
// 用户历史语言模型：整句（Poet）打分时给用户常用词/词对加分。
// unigram 取 userdb 词条自带的 commit_count；bigram 取本机历史文件
// （每行一次上屏：拼音\t词1\t词2…），按新旧分三池加权计数。
//
#ifndef RIME_USER_HISTORY_LM_H_
#define RIME_USER_HISTORY_LM_H_

#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <unordered_map>
#include <rime/common.h>

namespace rime {

class Config;
struct DictEntry;

struct HistoryLmParams {
  double unigram_weight = 0.0;
  double unigram_kappa = 4.0;
  double unigram_cap = 5.0;
  int unigram_min_length = 2;
  double bigram_weight = 0.0;
  double bigram_kappa = 1.0;
  double bigram_cap = 6.0;
  int bigram_min_count = 2;
  int priority_min_commits = 0;
  double priority_min_weight = -std::numeric_limits<double>::infinity();

  bool enabled() const { return unigram_weight > 0 || bigram_weight > 0; }
};

class UserHistoryLm {
 public:
  static constexpr std::array<size_t, 3> kPoolSizes = {256, 2048, 8192};
  static constexpr std::array<double, 3> kPoolWeights = {1.0, 0.5, 0.25};
  static constexpr size_t kCapacity = 256 + 2048 + 8192;

  using Counts = std::array<uint16_t, 3>;
  using SuccessorMap = std::unordered_map<string, Counts>;

  // 空路径 = 只在内存
  explicit UserHistoryLm(const string& file_path = string());

  // 同一路径进程内共享一份（script_translator 读、T9 上屏写）
  static an<UserHistoryLm> Get(const string& file_path);

  void Load();
  void Record(const string& pinyin, const vector<string>& words);
  void Clear();

  const SuccessorMap* Successors(const string& prev) const;
  // (加权计数, 原始计数)
  std::pair<double, int> Lookup(const string& prev, const string& word) const;

  uint64_t version() const { return version_; }
  size_t size() const;

 private:
  struct Line {
    string pinyin;
    vector<string> words;
  };
  void Add(Line line);
  void Count(const Line& line, size_t pool, int delta);
  void Rewrite();

  string file_path_;
  std::array<std::deque<Line>, 3> pools_;  // front = 最新
  std::unordered_map<string, SuccessorMap> bigrams_;
  uint64_t version_ = 0;
};

struct HistoryLmConfig {
  HistoryLmParams params;
  bool record = false;
  string file = "baiwu_history.txt";

  bool active() const { return record || params.enabled(); }
};

// 读 <ns>/history_lm/*；优先级门槛与 <ns>/user_phrase_priority_min_* 共用
HistoryLmConfig LoadHistoryLmConfig(Config* config, const string& name_space);

double HistoryBonus(const HistoryLmParams& params,
                    const UserHistoryLm::SuccessorMap* successors,
                    const DictEntry& entry);

// 句首上文：≤4 字的上文当一个词，否则不用
string HistoryStartWord(const string& preceding_text);

}  // namespace rime

#endif  // RIME_USER_HISTORY_LM_H_
