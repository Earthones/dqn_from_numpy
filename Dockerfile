# 使用轻量级的 Python 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件（如果有的话）并安装
COPY . .

# 暴露运行端口（虽然是 Hello World，但这是标准做法）
EXPOSE 8080

# 运行程序
CMD ["python", "app.py"]
