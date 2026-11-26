#ifndef RIME_USER_PHRASE_PRIORITY_H_
#define RIME_USER_PHRASE_PRIORITY_H_

namespace rime {

// 同码长时用户词默认绝对优先于系统词。min_commits / min_weight 给这个
// 优先权加门槛：只上屏过一次、或权重已衰减到接近 0 的用户词改按权重
// 与系统词比较（九键一串数字对应大量读音，偶然上屏的词不该霸占首选）。
// weight 传不含拼写惩罚的权重：模糊音匹配到的整段用户词照样保有优先权。
// 默认 (0, -inf) 即上游行为。
inline bool UserPhraseKeepsPriority(int commits,
                                    double weight,
                                    int min_commits,
                                    double min_weight) {
  return commits >= min_commits && weight >= min_weight;
}

}  // namespace rime

#endif  // RIME_USER_PHRASE_PRIORITY_H_
