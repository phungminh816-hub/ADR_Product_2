# Báo cáo tiến độ ADR_Product2

**Phạm vi:** dự đoán phản ứng có hại của thuốc bằng MLP và Random Forest.  
**Ngày cập nhật:** 08/11/2026.

## 1. Những việc đã hoàn thành

- Hoàn thiện các bước tải dữ liệu, làm sạch, chia tập, huấn luyện, đánh giá và nạp model để dự đoán từ SMILES (chuỗi biểu diễn phân tử).
- Xử lý được nhãn khuyết. Các bản ghi cùng phân tử được ghép khi thông tin bổ sung cho nhau, còn nhãn mâu thuẫn làm chương trình dừng để kiểm tra.
- Chia dữ liệu theo khung cấu trúc phân tử. Các phân tử cùng khung được giữ trong một tập, giúp hạn chế đánh giá quá thuận lợi do cấu trúc giống nhau xuất hiện ở cả train và validation/test.
- Đã huấn luyện và lưu cả MLP và Random Forest trên SIDER. Mỗi lượt chạy có model, cấu hình, điểm đánh giá và bản sao dữ liệu đã dùng.
- Đã sửa các đường dẫn cũ để phù hợp với thư mục `D:\HUST_20261\Project1\ADR_Product2`.
- Đã bổ sung chức năng tải và xử lý OnSIDES. Phần dữ liệu này mới hoàn thành một phần, chưa huấn luyện trên toàn bộ OnSIDES.

## 2. Dữ liệu SIDER và cách chia mẫu

SIDER hiện có **1.427 mẫu phân tử**, mỗi mẫu có **27 nhãn ADR**. Một phân tử có thể mang nhiều nhãn.

Phân vùng scaffold, seed 42 đang dùng:

- **Train:** 1.141 mẫu để học.
- **Validation:** 143 mẫu để chọn cấu hình và model.
- **Test:** 143 mẫu dành cho đánh giá cuối.

Ba tập không có khung cấu trúc giao nhau. Phân tử được biểu diễn bằng Morgan fingerprint 2.048 bit, radius 2 và giữ thông tin lập thể.

## 3. MLP: kết quả đã có và cấu hình chạy tiếp

**Model đã huấn luyện đang lưu dùng 128 nút ẩn**, có 265.755 tham số. Lượt chạy này học 20 epoch và giữ model tốt nhất ở epoch 12. Epoch là một lượt học qua tập train.

**Cấu hình chọn để chạy tiếp giữ 256 nút ẩn:** 2.048 đầu vào, 256 nút ẩn và 27 đầu ra. Cấu hình này có **531.483 tham số**. Nhiều nút hơn cho phép học nhiều tổ hợp đặc trưng hơn, nhưng phải kiểm tra bằng kết quả validation để biết có tốt hơn hay không.

Batch size đang dùng là **32 phân tử**. Với 1.141 mẫu train, mỗi epoch gồm 35 batch đủ 32 mẫu và một batch cuối có 21 mẫu, tổng cộng 36 lần cập nhật trọng số.

**Chưa có kết quả huấn luyện mới cho MLP 256 nút trong project này.** Không dùng điểm của bản 128 nút để đại diện cho bản 256 nút.

## 4. Random Forest

Random Forest học một bộ phân loại cho mỗi nhãn. Với 27 nhãn SIDER và **200 cây mỗi nhãn**, model hiện tại có **5.400 cây**. Độ sâu tối đa là 20, mỗi lá cần ít nhất hai mẫu.

Model chỉ dùng các mẫu có nhãn quan sát được ở đầu ra đang học. Random Forest không dùng batch size kiểu cập nhật trọng số của MLP.

## 5. Kết quả đã đạt được

Hai model dùng cùng phân vùng scaffold và được đánh giá trên 143 mẫu validation:

- **MLP 128 nút:** Macro AP **0,6518**; ROC-AUC **0,6304**; Micro F1 **0,8152**.
- **Random Forest:** Macro AP **0,6633**; ROC-AUC **0,6345**; Micro F1 **0,8211**.

Trong lượt chạy này, Random Forest đạt điểm cao hơn MLP ở cả ba chỉ số. Macro AP đánh giá khả năng xếp trường hợp dương lên cao. ROC-AUC đánh giá khả năng phân biệt dương và âm. Micro F1 cân bằng giữa tìm đúng và tìm đủ nhãn dương. Các điểm này không phải tần suất phản ứng trên bệnh nhân.

Cả hai run **chưa đánh giá tập test**. Chưa có cơ sở kết luận về MLP 256 nút hoặc chất lượng mô hình trên OnSIDES.

## 6. OnSIDES: lợi ích và phần đã làm

OnSIDES bổ sung một nguồn dữ liệu mới hơn: bản v3.1.1 cập nhật nhãn thuốc đến tháng 4/2026, trong khi SIDER 4.1 phát hành năm 2015. Phần xử lý hiện tại còn cho phép tạo nhãn phản ứng cụ thể như buồn nôn hoặc chóng mặt, thay vì chỉ dùng 27 nhóm rộng của SIDER trong project.

Dữ liệu gốc OnSIDES v3.1.1 có **1.866 hoạt chất**. Đây chưa phải số mẫu dùng để train vì còn phải ghép SMILES, lọc cấu trúc, loại thuốc phối hợp và gộp trùng.

Trạng thái hiện tại:

- Đã tải các bảng dữ liệu và kiểm tra mã xác minh SHA-256.
- Đã có code ghép hoạt chất với SMILES qua ChEMBL và lưu từng kết quả để chạy tiếp khi gián đoạn.
- Đã tra **40/1.866 hoạt chất**, có **27 kết quả SMILES hợp lệ**.
- Chưa tạo bộ dữ liệu OnSIDES xử lý đầy đủ, chưa chốt số mẫu cuối cùng và chưa huấn luyện đầy đủ hai model trên bộ này.

Code hiện chỉ lấy nguồn US, phần phản ứng bất lợi và thuốc đơn hoạt chất. Nhãn OnSIDES đang khác 27 nhóm SIDER nên chưa ghép trực tiếp hai bộ. Nhãn 0 trong phần xử lý OnSIDES nghĩa là quan hệ chưa được ghi nhận, không khẳng định thuốc không gây phản ứng đó.

## 7. Công việc tiếp theo

1. Huấn luyện MLP 256 nút, batch 32 trên cùng phân vùng SIDER để đối chiếu công bằng với Random Forest.
2. Hoàn tất ghép SMILES, làm sạch và xác định số mẫu OnSIDES thực tế.
3. Huấn luyện MLP và Random Forest trên cùng phân vùng OnSIDES với cùng bộ nhãn.
4. Chốt cấu hình bằng validation rồi mới đánh giá tập test.

## Nguồn đối chiếu

- Code: [MLP](../train/Train_MLP.py), [Random Forest](../train/Train_Ranfor.py), [chia dữ liệu](../data/split/make_split.py), [xử lý OnSIDES](../data/raw/onsides.py).
- Kết quả: [MLP validation](../metric/mlp_20261106T080758_282116Z_seed42/val.json), [Random Forest validation](../metric/ranfor_20261106T080800_003843Z_seed42/val.json).
- Dữ liệu: [SIDER 4.1](https://sideeffects.embl.de/download/), [OnSIDES v3.1.1](https://github.com/tatonetti-lab/onsides/releases/tag/v3.1.1).
