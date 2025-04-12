import json

def convert_data():
    # Đọc file JSON chứa labels
    with open('datasets/label.json', 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    
    # Đọc file JSONL chứa queries
    query_data = {}
    with open('datasets/queries.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            query_item = json.loads(line)
            query_data[str(query_item['query_id'])] = query_item['query']
    
    # Khởi tạo list để lưu kết quả
    result = []
    
    # Duyệt qua từng cặp key-value trong data
    for query_id, labels in label_data.items():
        # Tạo dictionary mới cho mỗi query
        item = {
            "query": query_id,
            "query_content": query_data.get(query_id, ""),  # Thêm nội dung query
            "label": []
        }
        
        # Xử lý từng label
        for label in labels:
            # Tách label thành phần tên bệnh và phần còn lại
            disease_name = label.split('|')[0]
            # Thêm tên bệnh vào list label nếu chưa tồn tại
            if disease_name not in item["label"]:
                item["label"].append(disease_name)
        
        # Thêm item vào kết quả
        result.append(item)
    
    # Lưu kết quả vào file mới
    with open('converted_data.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("Conversion completed! Results saved to converted_data.json")

if __name__ == "__main__":
    convert_data()