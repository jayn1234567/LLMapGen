# 生成特殊标记的脚本
special_tokens = [f'<|{i}|>' for i in range(1, 257)]

# 打印生成的特殊标记
for token in special_tokens:
    print(token)

# 可选：将特殊标记保存到文件
with open(r'D:\Workspace\基模型\训练\生产数据集\rc_nn\special_token.txt', 'w') as f:
    for token in special_tokens:
        f.write(token + '\n')

print("特殊标记已生成并保存到 special_tokens.txt")
