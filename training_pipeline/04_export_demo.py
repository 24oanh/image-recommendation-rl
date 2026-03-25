import os
import shutil

def export_demo_data():
    """
    Script gom toàn bộ dữ liệu từ thư mục huấn luyện
    chuyển sang thư mục cho Zeppelin chạy Demo.
    """
    # 1. Định nghĩa đường dẫn tương đối
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # THƯ MỤC NGUỒN: Nơi chứa file sau khi train xong
    source_dir = os.path.join(base_dir, 'data', 'processed') 
    
    # THƯ MỤC ĐÍCH: Nơi Zeppelin sẽ đọc dữ liệu
    target_dir = os.path.join(base_dir, 'zeppelin_demo', 'demo_data')
    
    # Tạo thư mục đích nếu chưa có
    os.makedirs(target_dir, exist_ok=True)

    # 2. Danh sách các file BẮT BUỘC CẦN cho Zeppelin
    files_to_export = [
        'idx2item.json',
        'item_popularity.json',
        'image_features_512.npy',
        'candidate_pool.npy',
        'best_dqn.pth',
        'best_a2c.pth'
    ]
    
    # 3. Copy từng file
    success_count = 0
    for filename in files_to_export:
        src_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(target_dir, filename)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f" Copy thành công: {filename:<25} | {size_mb:>6.2f} MB")
            success_count += 1
        else:
            print(f" Không tìm thấy file {filename} trong {source_dir}!")

    print(f"\n Đã copy {success_count}/{len(files_to_export)} files vào thư mục: {target_dir}")

if __name__ == "__main__":
    export_demo_data()
