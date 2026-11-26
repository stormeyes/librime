//
// Copyright RIME Developers
// Distributed under the BSD License
//
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <utf8.h>
#include <rime/config.h>
#include <rime/dict/vocabulary.h>
#include <rime/gear/user_history_lm.h>
#include <rime/gear/user_phrase_priority.h>

namespace rime {

constexpr std::array<size_t, 3> UserHistoryLm::kPoolSizes;
constexpr std::array<double, 3> UserHistoryLm::kPoolWeights;
constexpr size_t UserHistoryLm::kCapacity;

static size_t utf8_length(const string& s) {
  return utf8::unchecked::distance(s.c_str(), s.c_str() + s.length());
}

UserHistoryLm::UserHistoryLm(const string& file_path)
    : file_path_(file_path) {}

// ponytail: 依赖 Kotlin 侧 rimeLock 串行访问，未加锁；多线程访问需加 mutex
an<UserHistoryLm> UserHistoryLm::Get(const string& file_path) {
  static map<string, weak<UserHistoryLm>> instances;
  if (auto lm = instances[file_path].lock())
    return lm;
  auto lm = New<UserHistoryLm>(file_path);
  lm->Load();
  instances[file_path] = lm;
  return lm;
}

void UserHistoryLm::Count(const Line& line, size_t pool, int delta) {
  for (size_t i = 1; i < line.words.size(); ++i) {
    auto& successors = bigrams_[line.words[i - 1]];
    auto& counts = successors[line.words[i]];
    counts[pool] = static_cast<uint16_t>(counts[pool] + delta);
    if (counts == Counts{}) {
      successors.erase(line.words[i]);
      if (successors.empty())
        bigrams_.erase(line.words[i - 1]);
    }
  }
}

void UserHistoryLm::Add(Line line) {
  Count(line, 0, 1);
  pools_[0].push_front(std::move(line));
  for (size_t i = 0; i < pools_.size(); ++i) {
    while (pools_[i].size() > kPoolSizes[i]) {
      Line old = std::move(pools_[i].back());
      pools_[i].pop_back();
      Count(old, i, -1);
      if (i + 1 < pools_.size()) {
        Count(old, i + 1, 1);
        pools_[i + 1].push_front(std::move(old));
      }
    }
  }
  ++version_;
}

static bool ParseLine(const string& text, string* pinyin,
                      vector<string>* words) {
  size_t tab = text.find('\t');
  if (tab == string::npos)
    return false;
  *pinyin = text.substr(0, tab);
  words->clear();
  size_t start = tab + 1;
  while (start <= text.size()) {
    size_t end = text.find('\t', start);
    if (end == string::npos)
      end = text.size();
    if (end > start)
      words->push_back(text.substr(start, end - start));
    start = end + 1;
  }
  return !words->empty();
}

void UserHistoryLm::Load() {
  if (file_path_.empty())
    return;
  std::ifstream in(file_path_);
  if (!in)
    return;
  std::deque<Line> lines;
  size_t total = 0;
  string text;
  while (std::getline(in, text)) {
    ++total;
    Line line;
    if (!ParseLine(text, &line.pinyin, &line.words))
      continue;
    lines.push_back(std::move(line));
    if (lines.size() > kCapacity)
      lines.pop_front();
  }
  for (auto& line : lines)
    Add(std::move(line));
  if (total > kCapacity + kCapacity / 2)
    Rewrite();
}

void UserHistoryLm::Rewrite() {
  string tmp = file_path_ + ".tmp";
  {
    std::ofstream out(tmp, std::ios::trunc);
    for (size_t i = pools_.size(); i-- > 0;) {
      for (auto it = pools_[i].rbegin(); it != pools_[i].rend(); ++it) {
        out << it->pinyin;
        for (const auto& w : it->words)
          out << '\t' << w;
        out << '\n';
      }
    }
    if (!out)
      return;
  }
  std::rename(tmp.c_str(), file_path_.c_str());
}

static bool Sanitary(const string& s) {
  return s.find_first_of(string("\t\n\r\0", 4)) == string::npos;
}

void UserHistoryLm::Record(const string& pinyin, const vector<string>& words) {
  if (words.empty() || !Sanitary(pinyin) ||
      !std::all_of(words.begin(), words.end(),
                   [](const string& w) { return !w.empty() && Sanitary(w); }))
    return;
  Add(Line{pinyin, words});
  if (file_path_.empty())
    return;
  if (FILE* f = std::fopen(file_path_.c_str(), "a")) {
    std::fputs(pinyin.c_str(), f);
    for (const auto& w : words) {
      std::fputc('\t', f);
      std::fputs(w.c_str(), f);
    }
    std::fputc('\n', f);
    std::fclose(f);
  }
}

void UserHistoryLm::Clear() {
  for (auto& pool : pools_)
    pool.clear();
  bigrams_.clear();
  ++version_;
  if (!file_path_.empty())
    std::remove(file_path_.c_str());
}

const UserHistoryLm::SuccessorMap* UserHistoryLm::Successors(
    const string& prev) const {
  if (prev.empty())
    return nullptr;
  auto it = bigrams_.find(prev);
  return it == bigrams_.end() ? nullptr : &it->second;
}

static std::pair<double, int> Weigh(const UserHistoryLm::Counts& c) {
  double eff = 0;
  int raw = 0;
  for (size_t i = 0; i < c.size(); ++i) {
    eff += c[i] * UserHistoryLm::kPoolWeights[i];
    raw += c[i];
  }
  return {eff, raw};
}

std::pair<double, int> UserHistoryLm::Lookup(const string& prev,
                                             const string& word) const {
  auto* successors = Successors(prev);
  if (!successors)
    return {0, 0};
  auto it = successors->find(word);
  return it == successors->end() ? std::make_pair(0.0, 0) : Weigh(it->second);
}

size_t UserHistoryLm::size() const {
  size_t n = 0;
  for (const auto& pool : pools_)
    n += pool.size();
  return n;
}

double HistoryBonus(const HistoryLmParams& params,
                    const UserHistoryLm::SuccessorMap* successors,
                    const DictEntry& entry) {
  double bonus = 0;
  if (params.unigram_weight > 0 && entry.commit_count > 0 &&
      entry.credibility >= -0.01 &&
      UserPhraseKeepsPriority(entry.commit_count,
                              entry.weight - entry.credibility,
                              params.priority_min_commits,
                              params.priority_min_weight) &&
      utf8_length(entry.text) >= (size_t)params.unigram_min_length) {
    bonus += (std::min)(params.unigram_cap,
                        params.unigram_weight *
                            std::log1p(entry.commit_count /
                                       params.unigram_kappa));
  }
  if (params.bigram_weight > 0 && successors) {
    auto it = successors->find(entry.text);
    if (it != successors->end()) {
      auto [eff, raw] = Weigh(it->second);
      if (raw >= params.bigram_min_count)
        bonus += (std::min)(params.bigram_cap,
                            params.bigram_weight *
                                std::log1p(eff / params.bigram_kappa));
    }
  }
  return bonus;
}

HistoryLmConfig LoadHistoryLmConfig(Config* config, const string& name_space) {
  HistoryLmConfig c;
  if (!config)
    return c;
  const string ns = name_space + "/history_lm/";
  auto& p = c.params;
  config->GetDouble(ns + "unigram_weight", &p.unigram_weight);
  config->GetDouble(ns + "unigram_kappa", &p.unigram_kappa);
  config->GetDouble(ns + "unigram_cap", &p.unigram_cap);
  config->GetInt(ns + "unigram_min_length", &p.unigram_min_length);
  config->GetDouble(ns + "bigram_weight", &p.bigram_weight);
  config->GetDouble(ns + "bigram_kappa", &p.bigram_kappa);
  config->GetDouble(ns + "bigram_cap", &p.bigram_cap);
  config->GetInt(ns + "bigram_min_count", &p.bigram_min_count);
  config->GetInt(name_space + "/user_phrase_priority_min_commits",
                 &p.priority_min_commits);
  config->GetDouble(name_space + "/user_phrase_priority_min_weight",
                    &p.priority_min_weight);
  config->GetBool(ns + "record", &c.record);
  config->GetString(ns + "file", &c.file);
  return c;
}

string HistoryStartWord(const string& preceding_text) {
  size_t n = utf8_length(preceding_text);
  return n > 0 && n <= 4 ? preceding_text : string();
}

}  // namespace rime
