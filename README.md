# ADR_Product2

Huấn luyện XGBoost với Asymmetric Loss (ASL) và Random Forest để dự đoán đa nhãn ADR từ SMILES. Morgan fingerprint biểu diễn phân tử bằng 2.048 bit, bán kính 2 và giữ thông tin lập thể. Dữ liệu mặc định là SIDER do DeepChem phân phối, gồm 27 nhóm nhãn.

## Bộ benchmark

SIDER có 1.427 phân tử. Bản chia `scaffold_42` gồm 1.141 mẫu train, 143 validation và 143 test, với số scaffold giao nhau bằng 0 giữa từng cặp tập. Khi đã có bản chia, bắt đầu từ mục Train. Khi chỉ có `data/proceed/sider/processed.csv` và `labels.json`, bắt đầu từ mục Chia dữ liệu.

Benchmark chính dùng SIDER (`scaffold_42`, 27 nhóm ADR). Thí nghiệm mở rộng dùng OnSIDES (`onsides_42`, 100 phản ứng cụ thể). Mỗi bộ chạy XGBoost và Random Forest trên cùng một phân vùng scaffold Bemis–Murcko.

## Cấu trúc

```text
ADR_Product2/
├── data/
│   ├── raw/
│   │   ├── sider/            Dữ liệu thô SIDER (sider.csv, source.json)
│   │   ├── onside/           Dữ liệu thô OnSIDES
│   │   ├── download.py       Tải dữ liệu và ghi checksum nguồn
│   │   └── onsides.py        Tải và chuyển đổi OnSIDES
│   ├── proceed/
│   │   ├── sider/
│   │   ├── onsides/
│   │   └── prepare.py        Kiểm tra phân tử, nhãn, trùng và nhãn khuyết
│   └── split/
│       ├── scaffold_42/      Train, validation và test của SIDER
│       ├── onsides_42/       Train, validation và test của OnSIDES
│       └── make_split.py     Chia random hoặc scaffold và ghi thống kê
├── metric/                   Model, cấu hình và kết quả theo từng run
│   ├── sider/
│   │   ├── xgboost_res_sider/ Các lượt chạy XGBoost trên SIDER
│   │   └── ranfor_res_sider/ Các lượt chạy Random Forest trên SIDER
│   └── onside/
│       ├── xgboost_res_onside/ Các lượt chạy XGBoost trên OnSIDES
│       └── ranfor_res_onside/Các lượt chạy Random Forest trên OnSIDES
├── plot/
│   └── plot_results.py       Vẽ so sánh tổng thể và chênh lệch AP từng nhãn
├── train/
│   ├── common.py            Đặc trưng, kiểm tra split, metric và metadata
│   ├── runs.py              Tìm thư mục model và lượt chạy đã hoàn tất
│   ├── Train_Ranfor.py      Train Random Forest riêng cho từng nhãn
│   ├── Train_XGBoost.py     Train XGBoost riêng cho từng nhãn
│   ├── asymmetric_loss.py  Gradient và Hessian của ASL
│   ├── evaluate_test.py    Test checkpoint trên cùng split và xuất biểu đồ
│   └── predict.py           Dự đoán SMILES bằng model đã lưu
├── requirements.txt
└── README.md
```

Tên bộ dữ liệu trên CLI là `sider` và `onsides`. Đối với OnSIDES, dữ liệu nguồn nằm ở `data/raw/onside/`, bảng đã chuẩn hóa ở `data/proceed/onsides/` và kết quả train ở `metric/onside/`. Mỗi lượt train tạo thư mục mới. Các thư mục split và model được sinh trên máy chạy.

## Requirements và cài đặt

Mở PowerShell tại thư mục `ADR_Product2`. Dự án dùng Python từ 3.10 với bộ thư viện tương thích. Môi trường đã kiểm tra trên Windows dùng Python 3.14.7. Khi cài lần đầu, chạy:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
```

## 1. Tải dữ liệu

```powershell
python -m data.raw.download
```

Script dùng urllib để tải https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz, giải nén thành `data/raw/sider/sider.csv` và ghi URL, thời gian UTC cùng SHA-256 của bản nén và CSV vào `source.json`. Bản nén chỉ được lưu tạm khi tải. Dùng `--force` khi chủ động muốn thay dữ liệu raw.

## 2. Lọc dữ liệu

```powershell
python -m data.proceed.prepare
```

Đầu ra trong data/proceed/sider gồm processed.csv, labels.json, report.json. SMILES lỗi bị loại và được ghi vào báo cáo. Các phân tử trùng canonical SMILES được gộp khi nhãn tương thích, kể cả nhãn khuyết bổ sung nhau. Nhãn mâu thuẫn làm dừng xử lý. SMILES được canonicalize, giữ nguyên muối và dạng tautomer. Nhãn khuyết giữ NaN trong CSV.

Với CSV khác đã được chuẩn hóa thành smiles và các cột nhãn 0/1/NaN, truyền --input và --labels khi có thêm cột metadata. Dùng --output để đặt tên dataset mới.

## 3. Chia dữ liệu

```powershell
python -m data.split.make_split --strategy scaffold --seed 42
```

Chỉ tạo bản chia random khi cần thí nghiệm riêng. Với SIDER, chạy `python -m data.split.make_split --strategy random --seed 42`. Với OnSIDES, thêm `--dataset onsides`. Random Forest hỗ trợ cả hai chiến lược. Lệnh train XGBoost và lệnh so sánh test yêu cầu scaffold có số khung giao nhau bằng 0.

Mỗi chiến lược tạo thư mục riêng, chứa train.csv, val.csv, test.csv, labels.json và split.json. Tỷ lệ mục tiêu là 80/10/10; đổi bằng --train-ratio và --val-ratio. Mặc định train dùng scaffold_42. Đổi seed tạo phân vùng khác. Với scaffold, cả nhóm Bemis–Murcko được giữ trong cùng tập nên tỷ lệ dòng thực tế xấp xỉ mục tiêu. Toàn bộ scaffold rỗng cùng một nhóm. split.json ghi số mẫu, dương/âm/khuyết theo nhãn và số scaffold giao nhau giữa các tập. Random split có thể có scaffold trùng giữa các tập.

Chuẩn hóa và gộp trùng trước split là phép xử lý xác định từ cấu trúc và nhãn. Các thống kê học như trọng số nhãn chỉ tính từ train.

## 4. Train

```powershell
python -m train.Train_XGBoost --split data/split/scaffold_42
python -m train.Train_Ranfor --split data/split/scaffold_42
```

Cấu hình benchmark dùng seed 42:

```powershell
python -m train.Train_XGBoost --split data/split/scaffold_42 --rounds 160 --max-depth 5 --lr 0.06 --min-child-weight 0.5 --colsample-bytree 0.5
python -m train.Train_Ranfor --split data/split/scaffold_42 --trees 300 --balanced --max-features 0.05
python -m train.Train_XGBoost --split data/split/onsides_42 --rounds 150 --max-depth 3 --lr 0.07 --min-child-weight 0.5 --colsample-bytree 0.6
python -m train.Train_Ranfor --split data/split/onsides_42 --trees 300 --balanced --max-features 0.05
```

XGBoost mặc định huấn luyện một booster nhị phân cho mỗi nhãn, với 100 vòng, độ sâu tối đa 4 và learning rate 0,1. `--min-child-weight` đặt ngưỡng tổng Hessian ở nút con. `--colsample-bytree` đặt tỷ lệ đặc trưng lấy mẫu cho mỗi cây. ASL dùng `gamma_pos=1`, `gamma_neg=4` và `clip=0.05` để giảm đóng góp của các mẫu âm dễ. Objective tính đạo hàm theo raw margin và đặt sàn dương cho Hessian của mẫu đang đóng góp gradient. Mỗi booster bỏ qua nhãn khuyết của mình. Nhãn chỉ có một lớp train dùng điểm hằng. Cấu hình `--gamma-pos 0 --gamma-neg 0 --clip 0` cho loss BCE để đối chiếu ảnh hưởng của ASL trên validation.

Random Forest mặc định có 200 cây cho mỗi nhãn, max_depth=20, min_samples_leaf=2 và max_features=sqrt. Từng forest học các hàng có nhãn quan sát được ở đầu ra tương ứng và dùng dự đoán hằng với nhãn train chỉ có một lớp. `--max-depth 0` cho phép cây không giới hạn. `--balanced` dùng balanced_subsample. `--max-features` nhận tỷ lệ đặc trưng thử tại mỗi nút, ví dụ `0.06`.

## 5. Kết quả và test

Mỗi lần train tạo `metric/<dataset>/<model>_res_<dataset>/<model>_<UTC>_seed<seed>` với các file:

- config.json, labels.json, split.json và metadata.json để đối chiếu cấu hình và phiên bản thư viện.
- split_snapshot/ giữ chính xác dữ liệu của run, kèm checksum để kiểm tra.
- train.json và val.json chứa macro AP, macro ROC-AUC, micro precision/recall/F1 tại ngưỡng 0,5 và AP/ROC-AUC từng nhãn.
- XGBoost lưu `boosters.json` và checkpoint UBJSON trong `boosters/`. Random Forest lưu `model.joblib`. Cả hai lưu `summary.json`.

AP loại nhãn không có positive; ROC-AUC loại nhãn thiếu một lớp. Metric ghi rõ số nhãn hợp lệ, số dương/âm/khuyết và null cho điểm không xác định. Điểm AP và score dự đoán phục vụ đánh giá benchmark.

Sau khi chốt cấu hình bằng validation, chạy lệnh sau để nạp lại hai checkpoint, đánh giá trên cùng tập test và xuất biểu đồ:

```powershell
python -m train.evaluate_test --dataset sider
python -m train.evaluate_test --dataset onsides
```

Lệnh chọn lượt hoàn tất mới nhất của mỗi mô hình. Dùng `--xgboost-run` và `--rf-run` để chọn thư mục cụ thể. Bộ đánh giá kiểm tra checksum của hai phân vùng và yêu cầu scaffold giữa các tập giao nhau bằng 0. Dùng cùng `--metric-dir` cho lệnh train và test để lưu đầu ra vào thư mục riêng.

Kết quả mặc định lưu trong `metric/sider/test_comparison/` hoặc `metric/onside/test_comparison/`. `comparison.json` chứa số liệu tổng thể, số liệu từng nhãn và chênh lệch XGBoost trừ Random Forest. `comparison.png` vẽ Macro AP và Macro ROC-AUC từ điểm dự đoán, cùng Micro F1 tại ngưỡng 0,5. `per_label_ap_delta.png` đánh dấu các nhãn có chênh lệch AP lớn nhất theo mỗi hướng, kèm số mẫu dương `n+`.

## 6. Dự đoán

Truyền thư mục được in sau khi train cho `--run`. Ví dụ sau chọn lượt XGBoost mới nhất trên SIDER:

```powershell
$run = Get-ChildItem metric/sider/xgboost_res_sider/*/metadata.json |
    Where-Object { (Get-Content $_.FullName -Raw | ConvertFrom-Json).status -eq 'complete' } |
    Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty Directory
python -m train.predict --run $run.FullName --smiles 'CC(=O)Nc1ccc(O)cc1'
```

Lệnh nạp model cùng cấu hình fingerprint và trả 5 nhóm có điểm cao nhất. Chỉ nạp checkpoint do bạn tạo hoặc từ nguồn tin cậy. metadata.json có status=complete khi toàn bộ lượt train và ghi kết quả đã hoàn thành.

## Chạy với OnSIDES

Chạy tại thư mục project. Dữ liệu OnSIDES v3.1.1 đã tải trong `data/raw/onside/`; lệnh tải sẽ bỏ qua nếu đã có.

```powershell
python -m data.raw.download --dataset onsides
python -m data.proceed.prepare --dataset onsides
python -m data.split.make_split --dataset onsides --strategy scaffold --seed 42
python -m train.Train_XGBoost --split data/split/onsides_42
python -m train.Train_Ranfor --split data/split/onsides_42
```

Bước prepare tự tra SMILES qua ChEMBL, lưu từng kết quả để chạy tiếp nếu bị gián đoạn. Đã tra đủ 1.866 hoạt chất: 1.319 có SMILES hợp lệ, 280 không có cấu trúc hợp lệ và 267 không khớp tên qua cách tra hiện tại. Sau khi lọc US/AR, thuốc đơn hoạt chất và gộp cấu trúc, dữ liệu có 671 mẫu và 1.240 nhãn PT. Split `onsides_42` dùng 100 nhãn chọn từ train, gồm 537 train, 67 validation và 67 test. Dữ liệu đã có thì bỏ qua các lệnh tải, prepare và split, bắt đầu từ train. Muốn dùng mapping có sẵn khác, thêm `--smiles-map DUONG_DAN_CSV` với hai cột `rxnorm_id,smiles` và đặt thư mục output mới.

Chỉ dùng thuốc đơn hoạt chất, nguồn US, phần Adverse Reactions và nhãn MedDRA PT (tên phản ứng). Nhãn 1 là quan hệ được trích xuất, 0 là chưa được ghi nhận trong dữ liệu; không phải âm tính lâm sàng. Bộ nhãn này khác 27 nhóm SIDER. `--max-labels 100` chọn tối đa 100 nhãn theo số dương trong train, chỉ giữ nhãn có cả hai lớp ở train.

Nguồn dữ liệu: [OnSIDES v3.1.1](https://github.com/tatonetti-lab/onsides/releases/tag/v3.1.1). Kết quả prepare nằm trong `data/proceed/onsides/`; cách lưu model và dự đoán giữ như các mục trên.

## 7. Trực quan hóa kết quả (Plot)

Sau khi train, có thể mở trực tiếp hai figure validation của một model trong PowerShell. Biến `$run` là thư mục XGBoost đã chọn ở mục Dự đoán. Đóng figure đầu để xem figure kế tiếp.

```powershell
python -m plot.plot_results --run $run.FullName --show
```

Dùng `--chart ranking` để xem các nhãn có Average Precision (AP) cao nhất, hoặc `--chart scatter` để đối chiếu AP với số mẫu dương. Tham số `--file` nhận `train.json`, `val.json` hoặc `test.json` khi có. XGBoost và Random Forest hiện lưu điểm tổng kết. Chế độ đường học theo epoch dành cho những lượt cũ có `history.json`.

Vẽ lại hai biểu đồ test từ báo cáo JSON:

```powershell
python -m plot.plot_results --test-report metric/sider/test_comparison/comparison.json
```

Mặc định hai ảnh được ghi cạnh báo cáo. `--output` chọn thư mục xuất khác và `--top` đặt số nhãn mỗi hướng trong biểu đồ chênh lệch AP.

Đọc kết quả validation mới nhất của mỗi mô hình và so sánh các lượt có cùng split. Mặc định lệnh quét `metric/` và xuất biểu đồ 300 DPI vào `metric/figures/`. Dùng `--metric-dir` để quét thư mục đầu ra riêng.

```powershell
python -m plot.plot_results
```


- `model_comparison.png` so sánh Macro AP, Macro ROC-AUC và Micro F1 validation giữa XGBoost và Random Forest trên từng bộ dữ liệu.
- `per_label_sider.png` và `per_label_onside.png` xếp hạng nhãn theo AP của XGBoost trên validation, kèm số mẫu dương của từng nhãn.
- `sample_vs_ap_sider.png` (SIDER) và `sample_vs_ap_scatter.png` (OnSIDES) biểu diễn số mẫu dương và AP của mỗi nhãn trên validation.
