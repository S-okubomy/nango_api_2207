use std::collections::{ HashSet, HashMap };

/// TF-IDFの計算
/// https://www.sejuku.net/blog/26420
#[derive(Debug)]
pub struct TfIdf {
    pub word_vec: Vec<String>,
    pub tf_idf_vec: Vec<Vec<f64>>,
}

impl TfIdf {
    pub fn get_tf_idf(docs: &Vec<Vec<String>>) -> Self {
        let mut tmp_words: Vec<String> = Vec::new();
        for doc in docs {
            for w in doc {
                tmp_words.push(w.to_string());
            }
        }
        let words: HashSet<String> = tmp_words.into_iter().collect();
        let mut word_vec: Vec<String> = words.iter().map(|s| s.to_string()).collect();
        word_vec.sort();
    
        let mut tf_idf_vec: Vec<Vec<f64>> = Vec::new();
        let n: usize = docs.len();
        for i in 0..n {
            tf_idf_vec.push(vec![]);
            let d = &docs[i];
            for word in &word_vec {
                tf_idf_vec[i].push(cal_tf_idf(word.to_string(), &d, &docs));
            }
        }
        Self { word_vec, tf_idf_vec }
    }

    pub fn predict(tfidf: TfIdf, docs: &Vec<Vec<String>>, trg: &Vec<String>) -> Vec<(usize, f64)> {
        let trg_val = get_sentence_tf_idf(&tfidf.word_vec, docs, trg);
        get_cos_sort_list(&trg_val, &tfidf.tf_idf_vec)
    }
}

#[allow(dead_code)]
fn get_cos_max_id(trg: &Vec<f64>, docs: &Vec<Vec<f64>>) -> (usize, f64) {
    let mut max_id: usize = 0;
    let mut max_cos: f64 = 0.0;
    for doc in docs {
        let now_id: usize = doc[0] as usize;
        let tf_idf_vec: Vec<f64> = (&doc[1..]).to_vec();
        // println!("{:?} : {:?}", now_id, tf_idf_vec);
        let now_cos: f64 = calc_cos(trg, &tf_idf_vec);
        if max_cos < now_cos {
            max_id = now_id;
            max_cos = now_cos;
        }
    }

    (max_id, max_cos)
}

fn get_cos_sort_list(trg: &Vec<f64>, docs: &Vec<Vec<f64>>) -> Vec<(usize, f64)> {
    let mut cos_vec: Vec<(usize, f64)> = Vec::new();
    for doc in docs {
        let now_id: usize = doc[0] as usize;
        let tf_idf_vec: Vec<f64> = (&doc[1..]).to_vec();
        let now_cos: f64 = calc_cos(trg, &tf_idf_vec);
        cos_vec.push((now_id, now_cos));
    }
    cos_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // cos類似度の大きい順に並び替え
    cos_vec
}

fn get_sentence_tf_idf(word_vec: &Vec<String>, docs: &Vec<Vec<String>>, trg: &Vec<String>) -> Vec<f64> {
    let mut tf_idf_vec: Vec<f64> = Vec::new();
    for word in word_vec {
        tf_idf_vec.push(cal_tf_idf(word.to_string(), trg, &docs));
    }
    tf_idf_vec
}

/// cos類似度
/// https://qiita.com/yonedaco/items/ef6fd0db2773f62b0f72
/// https://w3e.kanazawa-it.ac.jp/math/category/vector/henkan-tex.cgi?target=/math/category/vector/naiseki-wo-fukumu-kihonsiki.html
fn calc_cos(a_vec: &Vec<f64>, b_vec: &Vec<f64>) -> f64 {
    // 文章aのベクトル長
    let a_len: f64 = a_vec.iter().fold(0_f64, |acc, cur| acc + cur.powf(2.0)).sqrt();
    // 文書bのベクトル長
    let b_len: f64 = b_vec.iter().fold(0_f64, |acc, cur| acc + cur.powf(2.0)).sqrt();

    if a_len == 0_f64 || b_len == 0_f64 {
        return 0_f64
    }

    // aとbの内積計算
    let mut dot_product: f64 = 0.0;
    let n = a_vec.len();
    for i in 0..n {
        dot_product += a_vec[i] * b_vec[i];
    }
    // println!("{} : {}", a_len, b_len);
    dot_product / (a_len * b_len)
}


fn tf(trg: &String, d: &Vec<String>) -> f64 {
    str_count(trg, d) as f64 / d.len() as f64
}

fn idf(t: &String, docs: &Vec<Vec<String>>) -> f64 {
    let mut df: f64 = 0.0;
    for doc in docs {
        if doc.contains(&t) {
            df += 1.0_f64;
        }
    }
    ((docs.len() as f64) / df).ln() + 1.0_f64
}

fn cal_tf_idf(t: String, d: &Vec<String>, docs: &Vec<Vec<String>>) -> f64 {
    tf(&t, d) * idf(&t, docs)
}

fn str_count(trg: &String, d: &Vec<String>) -> usize {
    let mut letters: HashMap<&str, usize> = HashMap::new();
    for s in d {
        let cnt = letters.entry(s).or_insert(0);
        *cnt += 1;
    }
    *letters.get(trg.as_str()).unwrap_or(&0)
}

#[cfg(test)]
mod tests {
    use super::*; // モジュールの外側で定義したリソース使用

    const ADD_EPSILON: f64 = 2.2204460492503131E-5_f64;

    #[test]
    fn str_count_test1() {
        let d = vec!["猫", "小さい", "犬", "犬", "可愛い", "可愛い", "犬", "大きい"].iter().map(|s| s.to_string()).collect();
        assert_eq!(str_count(&"犬".to_string(), &d), 3);
        assert_eq!(str_count(&"猫".to_string(), &d), 1);
        assert_eq!(str_count(&"ギター".to_string(), &d), 0);
        assert_eq!(str_count(&"小さい".to_string(), &d), 1);
    }

    #[test]
    fn tf_test1() {
        let d = vec!["犬", "可愛い", "犬", "大きい"].iter().map(|s| s.to_string()).collect();
        assert_eq!(tf(&"可愛い".to_string(), &d), 0.25);
        assert_eq!(tf(&"可愛くない".to_string(), &d), 0.0);
        assert_eq!(tf(&"大きい".to_string(), &d), 0.25);
        assert_eq!(tf(&"小さい".to_string(), &d), 0.0);
        assert_eq!(tf(&"犬".to_string(), &d), 0.5);
        assert_eq!(tf(&"猫".to_string(), &d), 0.0);
        assert_eq!(tf(&"虫".to_string(), &d), 0.0);
    }

    #[test]
    fn idf_test1() {
        let docs: Vec<Vec<String>> = vec![
                vec!["犬", "可愛い", "犬", "大きい"].iter().map(|s| s.to_string()).collect(),
                vec!["猫", "小さい", "猫", "可愛い", "可愛い"].iter().map(|s| s.to_string()).collect(),
                vec!["虫", "小さい", "可愛くない"].iter().map(|s| s.to_string()).collect()
        ];

        assert!(judge_diff(idf(&"可愛い".to_string(), &docs), 1.405465));
        // println!("{}", idf("可愛くない", &docs));
        assert!(judge_diff(idf(&"可愛くない".to_string(), &docs), 2.098612));
        assert!(judge_diff(idf(&"大きい".to_string(), &docs), 2.098612));
        assert!(judge_diff(idf(&"小さい".to_string(), &docs), 1.405465));
        assert!(judge_diff(idf(&"犬".to_string(), &docs), 2.098612));
        assert!(judge_diff(idf(&"猫".to_string(), &docs), 2.098612));
        assert!(judge_diff(idf(&"虫".to_string(), &docs), 2.098612));
    }

    #[test]
    fn cal_tf_idf_test1() {
        let docs: Vec<Vec<String>> = vec![
                vec!["犬", "可愛い", "犬", "大きい"].iter().map(|s| s.to_string()).collect(),
                vec!["猫", "小さい", "猫", "可愛い", "可愛い"].iter().map(|s| s.to_string()).collect(),
                vec!["虫", "小さい", "可愛くない"].iter().map(|s| s.to_string()).collect()
        ];
        assert!(judge_diff(cal_tf_idf("可愛い".to_string(), &docs[0], &docs), 0.351366));
        assert!(judge_diff(cal_tf_idf("大きい".to_string(), &docs[1], &docs), 0.000000));
        assert!(judge_diff(cal_tf_idf("小さい".to_string(), &docs[1], &docs), 0.281093));
        assert!(judge_diff(cal_tf_idf("虫".to_string(), &docs[2], &docs), 0.699537));

    }

    #[test]
    fn get_tf_idf_test1() {
        let docs: Vec<Vec<String>> = vec![
            vec!["犬", "可愛い", "犬", "大きい"].iter().map(|s| s.to_string()).collect(),
            vec!["猫", "小さい", "猫", "可愛い", "可愛い"].iter().map(|s| s.to_string()).collect(),
            vec!["虫", "小さい", "可愛くない"].iter().map(|s| s.to_string()).collect()
        ];

        let exp_v_v: Vec<Vec<f64>> = vec![
            vec![0.351366, 0.00000, 0.524653, 0.000000, 1.049306, 0.000000, 0.000000],
            vec![0.562186, 0.000000, 0.000000, 0.281093, 0.000000, 0.839445, 0.000000],
            vec![0.000000, 0.699537, 0.000000, 0.468488, 0.000000, 0.000000, 0.69953]
        ];
        assert!(judge_vec_diff(TfIdf::get_tf_idf(&docs).tf_idf_vec, exp_v_v));
    }

    #[test]
    fn calc_cos_test1() {
        let a_vec: Vec<f64> = vec![0.351366, 0.00000, 0.524653, 0.000000, 1.049306, 0.000000, 0.000000];
        // let a_vec: Vec<f64> = vec![2.0, 3.0, 5.0];
        let b_vec: Vec<f64> = vec![0.351366, 0.00000, 0.524653, 0.000000, 1.049306, 0.000000, 0.000000];

        // a_vecとb_vecが等しい→ 1になることを確認
        assert_eq!(calc_cos(&a_vec, &b_vec), 1.0);
    }

    #[test]
    fn get_sentence_tf_idf_test1() {

        let word_vec: Vec<String> = vec!["猫", "小さい", "犬", "可愛い", "大きい", "虫", "可愛くない"].iter().map(|s| s.to_string()).collect();
        let docs: Vec<Vec<String>> = vec![
            vec!["犬", "可愛い", "犬", "大きい"].iter().map(|s| s.to_string()).collect(),
            vec!["猫", "小さい", "猫", "可愛い", "可愛い"].iter().map(|s| s.to_string()).collect(),
            vec!["虫", "小さい", "可愛くない"].iter().map(|s| s.to_string()).collect()
        ];
        let trg: Vec<String> = vec!["猫", "大さい","ギター", "猫", "可愛い"].iter().map(|s| s.to_string()).collect();

        let res = get_sentence_tf_idf(&word_vec, &docs, &trg);
        assert_eq!(res, vec![0.8394449154672441, 0.0, 0.0, 0.2810930216216329, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn get_cos_max_id_test1() {
        let trg: Vec<f64> = vec![0.351366, 0.00000, 0.524653, 0.000000, 1.049306, 0.000000, 0.000000];
        // index 0番目はid
        let docs: Vec<Vec<f64>> = vec![
            vec![0.0, 0.351366, 0.00000, 0.524653, 0.000000, 1.049306, 0.000000, 0.000000],
            vec![1.0, 0.562186, 0.000000, 0.000000, 0.281093, 0.000000, 0.839445, 0.000000],
            vec![2.0, 0.000000, 0.699537, 0.000000, 0.468488, 0.000000, 0.000000, 0.69953]
        ];
        assert_eq!(get_cos_max_id(&trg, &docs), (0, 1.0));
    }

    #[test]
    fn get_cos_max_id_test2() {
        let trg: Vec<f64> = vec![0.000000, 0.699537, 0.000000, 0.468488, 0.000000, 0.000000, 0.69953];
        // index 0番目はid
        let docs: Vec<Vec<f64>> = vec![
            vec![0.0, 0.351366, 0.00000, 0.524653, 0.000000, 1.049306, 0.000000, 0.000000],
            vec![1.0, 0.562186, 0.000000, 0.000000, 0.281093, 0.000000, 0.839445, 0.000000],
            vec![2.0, 0.000000, 0.699537, 0.000000, 0.468488, 0.000000, 0.000000, 0.69953]
        ];
        assert_eq!(get_cos_max_id(&trg, &docs), (2, 1.0));
    }

    #[test]
    fn get_cos_max_id_test3() {
        let trg: Vec<f64> = vec![0.482186, 0.000000, 0.000000, 0.301093, 0.000000, 0.839445, 0.000000];
        // index 0番目はid
        let docs: Vec<Vec<f64>> = vec![
            vec![0.0, 0.351366, 0.00000, 0.524653, 0.000000, 1.049306, 0.000000, 0.000000],
            vec![1.0, 0.562186, 0.000000, 0.000000, 0.281093, 0.000000, 0.839445, 0.000000],
            vec![2.0, 0.000000, 0.699537, 0.000000, 0.468488, 0.000000, 0.000000, 0.69953]
        ];
        assert_eq!(get_cos_max_id(&trg, &docs), (1, 0.9973736484404528));
    }

    fn judge_diff(res: f64, exp: f64) -> bool {
        let abs_diff = (exp - res).abs();
        abs_diff <= f64::EPSILON + ADD_EPSILON // 許容範囲を超えたらfalse
    }

    fn judge_vec_diff(res_v_v: Vec<Vec<f64>>, exp_v_v: Vec<Vec<f64>>) -> bool {
        let n = res_v_v.len();
        for i in 0..n {
            let res_v = &res_v_v[i];
            let exp_v = &exp_v_v[i];
            let w = res_v.len();
            for j in 0..w {
                if !judge_diff(res_v[j], exp_v[j]) {
                    println!("{} {}", res_v[j], exp_v[j]);
                    return false; // 許容範囲を超えたらfalse
                }
            }
        }
        true
    }
}