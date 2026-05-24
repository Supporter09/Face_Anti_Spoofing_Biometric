# Quy trình thu thập webcam eval

## Mục tiêu

Thu thập 45 sessions để đánh giá thực tế:

- 15 `live`
- 15 `print_photo`
- 15 `phone_screen_replay`

Tối thiểu cần 30 sessions nếu không đủ thời gian hoặc thiết bị.

## Danh mục và điều kiện

- `live`: người thật, dùng webcam thông thường.
- `print_photo`: ảnh in giấy A4, laser hoặc inkjet, đặt trước camera.
- `phone_screen_replay`: ảnh hiển thị trên màn hình điện thoại, gồm Android và iPhone nếu có.

## Điều kiện quay

Thay đổi điều kiện giữa các session để tạo diversity:

- Ánh sáng: bình thường, mờ, ngược sáng.
- Khoảng cách: 30cm, 50cm, 80cm.

Không cần đổi tên thư mục sau khi quay. Script tự tạo timestamp cho từng session.

## Cách chạy script

```bash
python scripts/collect_webcam_eval.py --category live --out data_collected/webcam_eval
```

Ví dụ thêm ghi chú:

```bash
python scripts/collect_webcam_eval.py --category phone_screen_replay --out data_collected/webcam_eval --notes "iPhone, ánh sáng mờ, 50cm"
```

## Naming convention

Thư mục được tạo tự động theo timestamp:

```text
data_collected/webcam_eval/<category>/<YYYYMMDD_HHMMSS>/
```

Không đổi tên thư mục để tránh lệch metadata.

## Lỗi thường gặp

- Camera bị chiếm bởi app khác: tắt Zoom, trình duyệt, hoặc app camera rồi chạy lại.
- Ánh sáng quá tối: tăng sáng hoặc chuyển sang điều kiện `mờ` có kiểm soát, không để mặt mất hoàn toàn.
- Khuôn mặt bị cắt: điều chỉnh khoảng cách để mặt nằm giữa khung.

## Checklist trước khi quay

- Camera sạch.
- Ánh sáng đủ để thấy rõ khuôn mặt hoặc vật spoof.
- Khuôn mặt hoặc ảnh nằm giữa khung.
- Chọn đúng `--category`.
- Không đổi tên thư mục sau khi quay.
