# ADR_Product2

Huấn luyện Random Forest và MLP để dự đoán đa nhãn ADR từ SMILES. Morgan fingerprint biểu diễn phân tử bằng 2.048 bit, radius 2 và giữ thông tin lập thể. Dữ liệu chạy mặc định là SIDER do DeepChem phân phối, gồm 27 nhóm nhãn.

## Bản chạy đã thực hiện

Đã tải SIDER qua script download, lọc 1.427 phân tử và tạo cả random_42 (1.141/142/144) lẫn scaffold_42 (1.141/143/143). Scaffold_42 có số scaffold giao nhau bằng 0 giữa từng cặp tập. Khi dùng dữ liệu đi kèm, bắt đầu từ mục Train; các bước tải/lọc/chia phục vụ tái tạo dữ liệu trong thư mục output mới.

Hai run đã hoàn thành trên scaffold_42:

- metric/mlp_20261106T080758_282116Z_seed42: 20 epoch, checkpoint epoch 12, validation macro AP 0,651784, 265.755 tham số.
- metric/ranfor_20261106T080800_003843Z_seed42: 27 forest, 200 cây mỗi nhãn, validation macro AP 0,663273.

Cả hai đều được nạp lại để dự đoán SMILES qua train.predict. Các run này chưa đánh giá test. Các kiểm tra kỹ thuật đã qua gồm cú pháp, RF nhãn một lớp, metric bỏ qua nhãn khuyết, gradient masked loss, gộp nhãn khuyết bổ sung và từ chối duplicate mâu thuẫn. Dữ liệu fixture kiểm tra đã được dọn.

## Cấu trúc

```text
ADR_Product2/
├── data/
│   ├── raw/
│   │   ├── download.py       Tải dữ liệu và ghi checksum nguồn
│   │   └── onsides.py        Tải và chuyển đổi OnSIDES
│   ├── proceed/
│   │   └── prepare.py        Kiểm tra phân tử, nhãn, trùng và nhãn khuyết
│   └── split/
│       └── make_split.py     Chia random hoặc scaffold và ghi thống kê
├── metric/                  Model, cấu hình và kết quả theo từng run
├── train/
│   ├── common.py            Đặc trưng, kiểm tra split, metric và metadata
│   ├── Train_Ranfor.py      Train Random Forest riêng cho từng nhãn
│   ├── Train_MLP.py         Train MLP, masked BCE và early stopping
│   └── predict.py           Dự đoán SMILES bằng model đã lưu
├── requirements.txt
└── README.md
```

Các file __init__.py cho phép chạy bằng python -m. Thư mục proceed giữ đúng tên yêu cầu. Dữ liệu sinh ra nằm trong proceed/sider và split/<chiến_lược>_<seed>. Các thư mục dữ liệu và run đã tồn tại được bảo vệ khỏi ghi đè; dùng tên output mới cho thực nghiệm khác.

## Requirements và cài đặt

Python >=3.10 với phiên bản RDKit, PyTorch và scikit-learn tương thích. Mở PowerShell tại thư mục ADR_Product2 và chạy bằng môi trường Python đã cài requirements.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
```

## 1. Tải dữ liệu

```powershell
python -m data.raw.download
```

Script dùng urllib để tải https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz, giải nén thành data/raw/sider.csv và ghi URL, thời gian UTC, SHA-256 vào source.json. File .part được dọn khi lỗi. Dùng --force khi chủ động muốn thay dữ liệu raw.

## 2. Lọc dữ liệu

```powershell
python -m data.proceed.prepare
```

Đầu ra trong data/proceed/sider gồm processed.csv, labels.json, report.json. SMILES lỗi bị loại và được ghi vào báo cáo. Các phân tử trùng canonical SMILES được gộp khi nhãn tương thích, kể cả nhãn khuyết bổ sung nhau. Nhãn mâu thuẫn làm dừng xử lý. SMILES được canonicalize, giữ nguyên muối và dạng tautomer. Nhãn khuyết giữ NaN trong CSV.

Với CSV khác đã được chuẩn hóa thành smiles và các cột nhãn 0/1/NaN, truyền --input và --labels khi có thêm cột metadata. Dùng --output để đặt tên dataset mới.

## 3. Chia dữ liệu

```powershell
python -m data.split.make_split --strategy scaffold --seed 42
python -m data.split.make_split --strategy random --seed 42
```

Mỗi chiến lược tạo thư mục riêng, chứa train.csv, val.csv, test.csv, labels.json và split.json. Tỷ lệ mục tiêu là 80/10/10; đổi bằng --train-ratio và --val-ratio. Mặc định train dùng scaffold_42. Đổi seed tạo phân vùng khác. Với scaffold, cả nhóm Bemis–Murcko được giữ trong cùng tập nên tỷ lệ dòng thực tế xấp xỉ mục tiêu. Toàn bộ scaffold rỗng cùng một nhóm. split.json ghi số mẫu, dương/âm/khuyết theo nhãn và số scaffold giao nhau giữa các tập. Random split có thể có scaffold trùng giữa các tập.

Chuẩn hóa và gộp trùng trước split là phép xử lý xác định từ cấu trúc và nhãn. Các thống kê học như trọng số nhãn chỉ tính từ train.

## 4. Train

```powershell
python -m train.Train_MLP --split data/split/scaffold_42
python -m train.Train_Ranfor --split data/split/scaffold_42
```

Các ví dụ thay cấu hình hoặc chia random:

```powershell
python -m train.Train_MLP --split data/split/random_42 --hidden 256 --dropout 0.2 --weight-decay 0 --epochs 50
python -m train.Train_MLP --positive-weight --lr 0.0003 --device cpu
python -m train.Train_Ranfor --trees 300 --max-depth 12 --min-leaf 3 --balanced
```

MLP mặc định dùng 128 nút ẩn, ReLU, dropout 0,4, Adam, learning rate 0,001, weight decay 0,0001, batch 32. Train tối đa 50 epoch, early stopping sau 8 epoch không tăng validation macro AP. Masked BCE bỏ vị trí nhãn khuyết; --positive-weight tính tỷ lệ âm/dương từ train và giới hạn 1–10. Checkpoint lưu theo validation macro AP cao nhất.

Random Forest mặc định có 200 cây cho mỗi nhãn, max_depth=20, min_samples_leaf=2 và max_features=sqrt. Với 27 nhãn, tổng cộng 5.400 cây. Từng forest chỉ học các hàng có nhãn quan sát được ở đầu ra tương ứng; vì vậy hỗ trợ nhãn khuyết. Nhãn chỉ có một lớp train cho dự đoán hằng. Nhãn hoàn toàn không có quan sát train làm trainer dừng. --max-depth 0 cho phép cây không giới hạn; --balanced dùng balanced_subsample. Thiết kế này khác Random Forest đa đầu ra dùng chung 300 cây ở ADR_MLP_Next, cần đo lại khi so sánh.

## 5. Kết quả và test

Mỗi lần train tạo metric/<model>_<UTC>_seed<seed> với các file:

- config.json, labels.json, split.json và metadata.json để đối chiếu cấu hình và phiên bản thư viện.
- split_snapshot/ giữ chính xác dữ liệu của run, kèm checksum để kiểm tra.
- train.json và val.json chứa macro AP, macro ROC-AUC, micro precision/recall/F1 tại ngưỡng 0,5 và AP/ROC-AUC từng nhãn.
- MLP có model.pt và history.json; Random Forest có model.joblib; cả hai có summary.json.

AP loại nhãn không có positive; ROC-AUC loại nhãn thiếu một lớp. Metric ghi rõ số nhãn hợp lệ, số dương/âm/khuyết và null cho điểm không xác định. Điểm AP và score dự đoán phục vụ đánh giá benchmark.

Sau khi chốt cấu hình bằng validation, thêm --evaluate-test vào lệnh train của cấu hình đó để ghi test.json. Mặc định chỉ đánh giá train và validation. Đây là quy trình chủ động giữ test, người chạy cần tránh điều chỉnh cấu hình dựa trên điểm test.

## 6. Dự đoán

Thay TEN_RUN bằng thư mục được in sau khi train:

```powershell
python -m train.predict --run metric/TEN_RUN --smiles 'CC(=O)Nc1ccc(O)cc1'
```

Lệnh nạp model cùng cấu hình fingerprint và trả 5 nhóm có điểm cao nhất. Chỉ nạp checkpoint do bạn tạo hoặc từ nguồn tin cậy. metadata.json có status=complete khi toàn bộ lượt train và ghi kết quả đã hoàn thành.

## Chạy với OnSIDES

Chạy tại thư mục project. Dữ liệu OnSIDES v3.1.1 đã tải trong `data/raw/onsides/`; lệnh tải sẽ bỏ qua nếu đã có.

```powershell
python -m data.raw.download --dataset onsides
python -m data.proceed.prepare --dataset onsides
python -m data.split.make_split --processed data/proceed/onsides --output data/split/onsides_42 --max-labels 100
python -m train.Train_MLP --split data/split/onsides_42
python -m train.Train_Ranfor --split data/split/onsides_42
```

Bước prepare tự tra SMILES qua ChEMBL, lưu từng kết quả để chạy tiếp nếu bị gián đoạn. Mapping hiện mới tra 40/1.866 hoạt chất; lần chạy đầu có thể mất hàng chục phút hoặc hơn. Muốn dùng mapping có sẵn, thêm `--smiles-map DUONG_DAN_CSV` với hai cột `rxnorm_id,smiles`.

Chỉ dùng thuốc đơn hoạt chất, nguồn US, phần Adverse Reactions và nhãn MedDRA PT (tên phản ứng). Nhãn 1 là quan hệ được trích xuất, 0 là chưa được ghi nhận trong dữ liệu; không phải âm tính lâm sàng. Bộ nhãn này khác 27 nhóm SIDER. `--max-labels 100` chọn tối đa 100 nhãn theo số dương trong train, chỉ giữ nhãn có cả hai lớp ở train.

Nguồn dữ liệu: [OnSIDES v3.1.1](https://github.com/tatonetti-lab/onsides/releases/tag/v3.1.1). Kết quả prepare nằm trong `data/proceed/onsides/`; cách lưu model và dự đoán giữ như các mục trên.
