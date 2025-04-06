from word2vec import Word2Vec
import matplotlib.pyplot as plt

# Vietnamese corpus
corpus = [
    "Hà Nội là thủ đô của Việt Nam với lịch sử hàng nghìn năm văn hiến.",
    "Thành phố Hồ Chí Minh là trung tâm kinh tế sôi động, nơi tập trung nhiều doanh nghiệp lớn.",
    "Văn hóa ẩm thực Việt Nam rất đa dạng với phở, bún chả, và nhiều món ăn truyền thống khác.",
    "Du lịch Việt Nam đang phát triển mạnh mẽ với những danh lam thắng cảnh như Vịnh Hạ Long, Hội An.",
    "Người Việt Nam nổi tiếng với lòng hiếu khách và tinh thần làm việc cần cù.",
    "Giáo dục tại Việt Nam ngày càng được cải thiện với nhiều trường đại học chất lượng.",
    "Thể thao là phần quan trọng trong đời sống của người dân Việt Nam, đặc biệt là bóng đá, một blv ở Đức chia sẻ",
    "Lịch sử Việt Nam ghi nhận nhiều chiến công hào hùng chống ngoại xâm.",
    "Công nghệ thông tin và truyền thông phát triển nhanh chóng tại các thành phố lớn.",
    "Nông nghiệp Việt Nam đóng vai trò quan trọng trong kinh tế, với sản xuất gạo và cà phê nổi tiếng."
]

# Initialize and train the model
model = Word2Vec(
    vector_size=100,
    window=2,
    epochs=100,
    learning_rate=0.007,
    documents=corpus
)

# Train the model
print("Training Word2Vec model...")
model.train()

# Visualize sentence embeddings
print("\nVisualizing sentence embeddings...")
model.visualize_sentences(title="Vietnamese Sentence Embeddings")

# Example search
query = "ẩm thực Việt Nam"
print(f"\nSearching for documents related to: '{query}'")
results = model.search(query, top_k=3)

print("\nTop 3 results:")
for i, result in enumerate(results):
    print(f"{i+1}. Score: {result['score']:.4f}")
    print(f"   Text: {result['text']}")
    print()

# Save the model
model.save_model("vietnamese_word2vec_model.pkl")
print("Model saved to 'vietnamese_word2vec_model.pkl'") 