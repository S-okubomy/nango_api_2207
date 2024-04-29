// use lambda_runtime::{service_fn, LambdaEvent, Error};
use serde_json::{json, Value};
use std::error::Error as OtherError;

use std::fs::File;
use vaporetto::{Model, Predictor, Sentence};
use vaporetto_rules::{
    string_filters::KyteaFullwidthFilter, StringFilter,
};

use std::collections::HashMap;

// use lambda_http::{service_fn, Error, IntoResponse, Request, RequestExt, Response};

use query_map::QueryMap;


mod nlp;
use nlp::tf_idf;

const STR_PKEY: &str = "nango7_ai_nango_kun";



use lambda_apigateway_response::{
    http::StatusCode,
    types::{
        Headers,
        MultiValueHeaders,
    },
    Response,
};
use lambda_runtime::{
    Error as LambdaError,
    LambdaEvent,
};



// async fn handler(
//     event: LambdaEvent<ApiGatewayProxyRequest>,
// ) -> Result<ApiGatewayProxyResponse, Error> {
//     let mut headers = HeaderMap::new();
//     headers.insert("content-type", "text/html".parse().unwrap());
//     let resp = ApiGatewayProxyResponse {
//         status_code: 200,
//         multi_value_headers: headers.clone(),
//         is_base64_encoded: Some(false),
//         body: Some("Hello AWS Lambda HTTP request".into()),
//         headers,
//     };
//     Ok(resp)
// }

// #[tokio::main]
// async fn main() -> Result<(), Error> {
//     lambda_runtime::run(service_fn(handler)).await
// }



type LambdaResult<T> = Result<T, LambdaError>;
 
async fn handler(
    event: LambdaEvent<serde_json::Value>,
) -> LambdaResult<Response<serde_json::Value>> {
    // let res = Response {
    //     status_code: StatusCode::OK,
    //     body: json!({
    //         "message": "Hello world!",
    //     }),
    //     headers: Headers::new(),
    //     multi_value_headers: MultiValueHeaders::new(),
    //     is_base64_encoded: true,
    // };


    // let (event, _context) = event.into_parts();
    // 入力パラメータを得る
    // let q_map = _event.query_string_parameters();

    let exec_mode: Result<ExecMode, String> = ExecMode::new(event);
    match exec_mode {
        Err(error) => {
            let message = format!("error running init: {}", error);
            let res_err_json: Value = json!({
                "success": false,
                "message": message,
            });

            // let mut headers = Headers::new();
            // headers.insert("content-type".to_string(), "application/json".parse().unwrap());
            // headers.insert("Access-Control-Allow-Methods".to_string(), "OPTIONS,POST,GET".parse().unwrap());
            // headers.insert("Access-Control-Allow-Credential".to_string(), "true".parse().unwrap());
            // headers.insert("Access-Control-Allow-Origin".to_string(), "*".parse().unwrap());

            let res = Response {
                status_code: StatusCode::BAD_REQUEST,
                body: res_err_json,
                headers: get_Header(),
                multi_value_headers: MultiValueHeaders::new(),
                is_base64_encoded: true,
            };
            Ok(res)


        },
        Ok(mode) => {
            let res_json: Value = run(mode);
            // let mut headers = Headers::new();
            // headers.insert("content-type".to_string(), "text/html".parse().unwrap());

            let res = Response {
                status_code: StatusCode::OK,
                body: res_json,
                headers: get_Header(),
                multi_value_headers: MultiValueHeaders::new(),
                is_base64_encoded: true,
            };

            Ok(res)
        }
    }


}

fn get_Header() -> HashMap<String, String> {
    let mut headers = Headers::new();
    headers.insert("content-type".to_string(), "application/json".parse().unwrap());
    headers.insert("Access-Control-Allow-Methods".to_string(), "OPTIONS,POST,GET".parse().unwrap());
    headers.insert("Access-Control-Allow-Credential".to_string(), "true".parse().unwrap());
    headers.insert("Access-Control-Allow-Origin".to_string(), "*".parse().unwrap());

    headers
}
 
#[tokio::main]
async fn main() -> LambdaResult<()> {
    let handler_fn = lambda_runtime::service_fn(handler);
    lambda_runtime::run(handler_fn).await?;
 
    Ok(())
}






/// 使用例
/// 学習時: {"mode": "l", "pkey": "nango7_ai_nango_kun"}
/// 予測時: {"mode": "p", "que_sentence": "お店で楽器は演奏できますか？", "pkey": "nango7_ai_nango_kun"}
// #[tokio::main]
// async fn main() -> Result<(), Error> {
//     lambda_http::run(service_fn(handler)).await
// }

// async fn handler(event: Request) -> Result<impl IntoResponse, Error> {

//     // let (event, _context) = event.into_parts();
//     // 入力パラメータを得る
//     let q_map = event.query_string_parameters();

//     let exec_mode: Result<ExecMode, String> = ExecMode::new(q_map);
//     match exec_mode {
//         Err(error) => {
//             let message = format!("error running init: {}", error);
//             let res_err_json: Value = json!({
//                 "code": 400,
//                 "success": false,
//                 "message": message,
//             });

//             let resp = Response::builder()
//                 .status(400)
//                 .header("Content-Type", "application/json")
//                 .body(res_err_json.to_string())
//                 .map_err(Box::new)?;
//             Ok(resp)
//         },
//         Ok(mode) => {
//             let res_json: Value = run(mode);
//             let resp = Response::builder()
//                 .status(200)
//                 .header("Content-Type", "application/json")
//                 .header("Access-Control-Allow-Methods", "OPTIONS,POST,GET")
//                 .header("Access-Control-Allow-Credential", "true")
//                 .header("Access-Control-Allow-Origin", "*")
//                 .body(res_json.to_string())
//                 .map_err(Box::new)?;
//             Ok(resp)
//         }
//     }
// }


#[derive(Debug)]
enum ExecMode {
    Learn,
    Predict { que_sentence: String },
}

impl ExecMode {
    fn new(event: LambdaEvent<serde_json::Value>) -> Result<ExecMode, String> {
        let (params, _context) = event.into_parts();

        let mode: &str = params["mode"].as_str().unwrap_or("");
        let que_sentence = params["que_sentence"].as_str().unwrap_or("");
        let pkey = params["pkey"].as_str().unwrap_or("");

        // let mode: &str = q_map.first("mode").unwrap_or("");
        // let que_sentence = q_map.first("que_sentence").unwrap_or("");
        // let pkey = q_map.first("pkey").unwrap_or("");

        // let mode: &str = "p";
        // let que_sentence = "test";
        // let pkey = "nango7_ai_nango_kun";

        if pkey.len() == 0 || pkey != STR_PKEY {
            return Err("Not executable".to_string());
        }

        match mode {
            "l" => {
                Ok(ExecMode::Learn)
            },
            "p" => {
                if que_sentence.len() > 0 {
                    Ok(ExecMode::Predict { que_sentence: que_sentence.to_string() })
                } else {
                    Err("予測時は、質問文を入力してください。".to_string())
                }
            },
            _ => {
                Err("学習: l、予測: p を指定してください。".to_string())
            }
        }
    }
}

fn run(mode: ExecMode) -> Value {
    match mode {
        ExecMode::Learn => {
            learn()
        },
        ExecMode::Predict { que_sentence } => {
            predict(que_sentence)
        },
    }
}

fn learn() -> Value {
    let qa_data: QaData = read_csv().unwrap_or_else(|err| {
        println!("error running read: {}", err);
        std::process::exit(1);
    });

    let mut docs: Vec<Vec<String>> = Vec::new();
    for input_qa in qa_data.que_vec {
        let doc_vec: Vec<String> = get_tokenizer(input_qa);
        docs.push(doc_vec);
    }

    out_csv_word(&docs).unwrap_or_else(|err| {
        println!("error running out_csv_word csv: {}", err);
        std::process::exit(1);
    });

    let tf_idf_res = tf_idf::TfIdf::get_tf_idf(&docs);
    // 学習済みモデル出力
    out_csv(tf_idf_res).unwrap_or_else(|err| {
        println!("error running output csv: {}", err);
        std::process::exit(1);
    });

    let res_json: Value = json!({
        "code": 200,
        "success": true,
        "mode": "learn",
    });
    res_json
}

fn predict(que_sentence: String) -> Value {
    let qa_data: QaData = read_csv().unwrap_or_else(|err| {
        println!("error running read: {}", err);
        std::process::exit(1);
    });

    let docs: Vec<Vec<String>> = read_word_list_csv().unwrap_or_else(|err| {
        println!("error running read: {}", err);
        std::process::exit(1);
    });

    let tfidf: tf_idf::TfIdf = read_model_csv().unwrap();
    let trg: Vec<String> = get_tokenizer(que_sentence.to_owned());
    let ans_vec: Vec<(usize, f64)> = tf_idf::TfIdf::predict(tfidf, &docs, &trg);

    let res_json: Value = make_json(que_sentence, qa_data, ans_vec);
    res_json
}


fn make_json(que_sentence: String, qa_data: QaData, ans_vec: Vec<(usize, f64)>) -> Value {
    let mut qa_infos: Vec<Value> = Vec::new();
    for (id, cos_val) in ans_vec {
        if cos_val > 0.3 {
            qa_infos.push(json!({
                "que": que_sentence,
                "ans": qa_data.ans_vec[id],
                "cos_val": cos_val,
                "similar_que": qa_data.que_vec[id]
            }));
        }
    }

    let res_json: Value = json!({
        "code": 200,
        "success": true,
        "mode": "predict",
        "payload": {
            "qa_infos": qa_infos
        }
    });
    res_json
}

fn get_tokenizer(doc: String) -> Vec<String> {
    let mut f = zstd::Decoder::new(File::open("./model/bccwj-luw-small.model.zst").unwrap()).unwrap();
    let model = Model::read(&mut f).unwrap();
    let predictor = Predictor::new(model, true).unwrap();

    let pre_filters: Vec<Box<dyn StringFilter<String>>> = vec![
        Box::new(KyteaFullwidthFilter),
    ];
    
    let preproc_input = pre_filters.iter().fold(doc, |s, filter| filter.filter(s));
    
    let mut sentence = Sentence::from_raw(preproc_input).unwrap();
    predictor.predict(&mut sentence);
    
    let mut buf = String::new();
    sentence.write_tokenized_text(&mut buf);
    // output the tokens
    let docs: Vec<String> = buf.split(" ").map(|s| s.to_string()).collect();
    // println!("{:?}", docs);

    docs
}

#[derive(Debug)]
struct QaData {
    que_vec: Vec<String>,
    ans_vec: Vec<String>,
}

fn read_csv() -> Result<QaData, Box<dyn OtherError>> {
    let csv_file_path = "input/study_qa1.csv";
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false) // ヘッダーが無い事を明示的に設定
        .from_path(csv_file_path)?;

    let mut que_vec: Vec<String> = Vec::new();
    let mut ans_vec: Vec<String> = Vec::new();
    for result in rdr.records() {
        let record = result?;
        que_vec.push(record[3].to_string());
        ans_vec.push(record[2].to_string())
    }
    Ok(QaData { que_vec, ans_vec })
}

fn read_word_list_csv() -> Result<Vec<Vec<String>>, Box<dyn OtherError>> {
    let csv_file_path = "output/word_list.csv";
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false) // ヘッダーが無い事を明示的に設定
        .flexible(true) // 可変長で読み込み
        .from_path(csv_file_path)?;

    let mut word_v_v: Vec<Vec<String>> = Vec::new();
    for (index, result) in rdr.records().enumerate() { // ヘッダーは除く
        let record = result?;
        word_v_v.push(vec![]);
        for col in &record {
            word_v_v[index].push(col.to_string());
        }
    }

    Ok(word_v_v)
}

fn read_model_csv() -> Result<tf_idf::TfIdf, Box<dyn OtherError>> {
    let model_csv_file_path = "output/model_qa1.csv";
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false) // ヘッダーが無い事を明示的に設定
        .from_path(model_csv_file_path)?;

    let mut rec_v_v: Vec<Vec<String>> = Vec::new();
    for (index, result) in rdr.records().enumerate() { // ヘッダーは除く
        let record = result?;
        rec_v_v.push(vec![]);
        for col in &record {
            rec_v_v[index].push(col.to_string());
        }
    }
    let word_vec: Vec<String> = (rec_v_v[0][1..]).to_vec(); // "id"の文字以降を格納
    let mut tf_idf_vec: Vec<Vec<f64>> = Vec::new();
    for (index, rec_v) in rec_v_v.iter().skip(1).enumerate() { // ヘッダーは除く
        tf_idf_vec.push(vec![]);
        for tf_idf in rec_v {
            let tf_idf_val: f64 = tf_idf.parse::<f64>().unwrap();
            tf_idf_vec[index].push(tf_idf_val);
        }
    }

    let tfidf: tf_idf::TfIdf = tf_idf::TfIdf {
        word_vec,
        tf_idf_vec
    };

    Ok(tfidf)
}

/// csv出力
/// https://qiita.com/algebroid/items/c456d4ec555ae04c7f92
fn out_csv(tf_idf_res: tf_idf::TfIdf) -> Result<(), Box<dyn OtherError>> {
    let csv_file_out_path = "output/model_qa1.csv";
    let mut wtr = csv::WriterBuilder::new()
        .quote_style(csv::QuoteStyle::Always)
        .from_path(csv_file_out_path)?;

    let mut w_vec = vec!["id"];
    let mut w_add_vec: Vec<&str> = tf_idf_res.word_vec.iter().map(|s| s.as_str()).collect();
    w_vec.append(&mut w_add_vec);
    wtr.write_record(&w_vec)?;

    for (index, tf_idf_vec) in tf_idf_res.tf_idf_vec.iter().enumerate() {
        let mut s_vec: Vec<String> = vec![index.to_string()];
        let mut s_add_vec: Vec<String> = tf_idf_vec.iter().map(|s| s.to_string()).collect();
        s_vec.append(&mut s_add_vec);
        wtr.write_record(s_vec)?;
    }

    wtr.flush()?;
    Ok(())
}

fn out_csv_word(docs: &Vec<Vec<String>>) -> Result<(), Box<dyn OtherError>> {
    let csv_file_out_path = "output/word_list.csv";
    let mut wtr = csv::WriterBuilder::new()
        .quote_style(csv::QuoteStyle::Always)
        .flexible(true) // 可変長で書き込み
        .from_path(csv_file_out_path)?;

    for doc in docs {
        let s_vec: Vec<String> = doc.iter().map(|s| s.to_string()).collect();
        wtr.write_record(s_vec)?;
    }

    wtr.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learn_test1() {
        let res = learn();
        // println!("{:?}", res.to_string());
        let exp: Value = json!({
            "code": 200,
            "success": true,
            "mode": "learn",
        });
        assert_eq!(res, exp);
    }

    #[test]
    fn predict_test1() {
        let que_sentence: String = "おすすめのメニュー教えてください。".to_string();
        let res = predict(que_sentence.to_owned());
        // println!("{} {} {}", res["code"], res["mode"], res["payload"]["qa_infos"][0]);
        let tmp_res_vec: Vec<String> = vec![&res["code"], &res["mode"], &res["payload"]["qa_infos"][0]["que"]]
            .into_iter().map(|v| v.to_string() ).collect();
        let res_vec: Vec<&str> = tmp_res_vec.iter().map(|s| s.as_str()).collect();
        let exp_que: String = "\"".to_string() + &que_sentence.as_str() + "\"";
        let exp_vec = vec!["200", "\"predict\"", exp_que.as_str()];
        assert_eq!(res_vec, exp_vec);
    }

    #[test]
    fn init_pkey_test1() {
        let event: Value = json!({
            "mode": "l", // pkeyがない場合にエラーとなるか確認
        });
        let res = ExecMode::new(event);
        match res {
            Err(error) => {
                assert_eq!(error, "Not executable".to_string());
            },
            Ok(_) => {
                assert!(false);
            }
        }
    }

    #[test]
    fn init_pkey_test2() {
        let event: Value = json!({
            "mode": "l",
            "pkey": "" // pkeyが不正な場合(空)、エラーとなるか確認
        });
        let res = ExecMode::new(event);
        match res {
            Err(error) => {
                assert_eq!(error, "Not executable".to_string());
            },
            Ok(_) => {
                assert!(false);
            }
        }
    }

    #[test]
    fn init_pkey_test3() {
        let event: Value = json!({
            "mode": "l",
            "pkey": "abc" // pkeyが不正な場合(間違い)、エラーとなるか確認
        });
        let res = ExecMode::new(event);
        match res {
            Err(error) => {
                assert_eq!(error, "Not executable".to_string());
            },
            Ok(_) => {
                assert!(false);
            }
        }
    }

    #[test]
    fn init_test1() {
        let event: Value = json!({
            "mode": "x", // 不正なモードでエラーとなるか確認
            "pkey": "nango7_ai_nango_kun"
        });
        let res = ExecMode::new(event);
        match res {
            Err(error) => {
                assert_eq!(error, "学習: l、予測: p を指定してください。".to_string());
            },
            Ok(_) => {
                assert!(false);
            }
        }
    }

    #[test]
    fn init_test2() {
        let event: Value = json!({
            "mode": "l", // 学習モードで処理実行されるか確認
            "pkey": "nango7_ai_nango_kun",
        });
        let res = ExecMode::new(event);
        match res {
            Err(_) => {
                assert!(false);
            },
            Ok(_) => {
                assert!(true);
            }
        }
    }

    #[test]
    fn init_test3() {
        let event: Value = json!({
            "mode": "p", // 類推モードで処理実行されるか確認
            "que_sentence": "お店で楽器は演奏できますか？",
            "pkey": "nango7_ai_nango_kun",
        });
        let res = ExecMode::new(event);
        match res {
            Err(_) => {
                assert!(false);
            },
            Ok(_) => {
                assert!(true);
            }
        }
    }

    #[test]
    fn init_test4() {
        let event: Value = json!({
            "mode": "p", // 類推モードで処理実行されるか確認
            "que_sentence": "", // 質問文が未入力時にエラーとなるか確認
            "pkey": "nango7_ai_nango_kun",
        });
        let res = ExecMode::new(event);
        match res {
            Err(error) => {
                assert_eq!(error, "予測時は、質問文を入力してください。".to_string());
            },
            Ok(_) => {
                assert!(false);
            }
        }
    }

}