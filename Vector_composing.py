import math
import numpy as np

def transform_vectors(A, B, C, use_degrees=True):
    """
    将三个向量A, B, C转换为新向量t，支持浮点数输入
    
    参数:
    A, B, C: 三个同维度的向量(列表或数组)，支持浮点数
    use_degrees: 是否将角度输出为度，False则输出弧度
    
    返回:
    t: 新向量，格式为 [r1, φ1, δ1, r2, φ2, δ2, ...]
    """
    # 检查维度是否一致
    if len(A) != len(B) or len(B) != len(C):
        raise ValueError("输入向量的维度必须相同")
    
    n = len(A)
    t = []
    
    # 使用小的epsilon值来处理浮点数精度问题
    epsilon = 1e-12
    
    for i in range(n):
        a, b, c = float(A[i]), float(B[i]), float(C[i])
        
        # 1. 计算模长 r_i
        r_i = math.sqrt(a*a + b*b + c*c)
        
        # 2. 计算方位角 φ_i (在A-B平面投影与A轴的夹角)
        # 使用atan2处理所有象限，并考虑浮点数精度
        if abs(a) < epsilon and abs(b) < epsilon:
            phi_i = 0.0  # 避免除以零，设默认值
        else:
            phi_i = math.atan2(b, a)
        
        # 3. 计算仰角 δ_i (与A-B平面的夹角)
        if r_i < epsilon:
            delta_i = 0.0  # 避免除以零
        else:
            # 计算在A-B平面上的投影长度
            projection_length = math.sqrt(a*a + b*b)
            # 仰角是与A-B平面的夹角
            if projection_length < epsilon:
                # 如果投影长度接近0，则点几乎在C轴上
                delta_i = math.copysign(math.pi/2, c)  # +π/2或-π/2
            else:
                delta_i = math.atan2(c, projection_length)
        
        # 转换为度（如果需要）
        if use_degrees:
            phi_i = math.degrees(phi_i)
            delta_i = math.degrees(delta_i)
        
        # 添加到结果向量
        t.extend([r_i, phi_i, delta_i])
    
    return t

# 使用numpy的向量化版本（更高效，特别适合浮点数）
def transform_vectors_numpy(A, B, C, use_degrees=True):
    """
    使用numpy的向量化实现，优化浮点数处理
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)
    
    # 检查维度
    if A.shape != B.shape or B.shape != C.shape:
        raise ValueError("输入向量的维度必须相同")
    
    # 设置小的epsilon值来处理浮点数精度
    epsilon = 1e-12
    
    # 计算模长
    r = np.sqrt(A**2 + B**2 + C**2)
    
    # 计算方位角
    phi = np.arctan2(B, A)
    # 处理A=B≈0的情况
    mask_phi = (np.abs(A) < epsilon) & (np.abs(B) < epsilon)
    phi = np.where(mask_phi, 0.0, phi)
    
    # 计算仰角
    projection_length = np.sqrt(A**2 + B**2)
    delta = np.arctan2(C, projection_length)
    # 处理投影长度接近0的情况
    mask_delta = projection_length < epsilon
    delta = np.where(mask_delta, np.copysign(np.pi/2, C), delta)
    # 处理r=0的情况
    delta = np.where(r < epsilon, 0.0, delta)
    
    # 转换为度（如果需要）
    if use_degrees:
        phi = np.degrees(phi)
        delta = np.degrees(delta)
    
    # 交错合并结果
    t = np.column_stack([r, phi, delta]).flatten()
    
    return t.tolist()

# 测试浮点数示例
if __name__ == "__main__":
    # 示例1：浮点数向量
    A1 = [1.5, 2.7, 3.14159]
    B1 = [4.2, 5.8, 6.0] 
    C1 = [7.1, 8.9, 9.99]
    
    print("示例1 - 浮点数向量:")
    result1 = transform_vectors(A1, B1, C1)
    print(f"输入: A={A1}, B={B1}, C={C1}")
    print(f"输出: {result1}")
    print()
    
    # 示例2：包含零和接近零的浮点数
    A2 = [1.0, 0.0, -1.5, 0.0000001]
    B2 = [0.0, 1.0, 0.0, -0.0000001]
    C2 = [0.0, 0.0, 1.0, 0.0]
    
    print("示例2 - 包含零和接近零的浮点数:")
    result2 = transform_vectors_numpy(A2, B2, C2)
    print(f"输入: A={A2}, B={B2}, C={C2}")
    print(f"输出: {result2}")
    print()
    
    # 示例3：使用弧度输出
    print("示例3 - 使用弧度输出:")
    result3 = transform_vectors(A1, B1, C1, use_degrees=False)
    print(f"输出(弧度): {result3}")
    print()
    
    # 验证两种方法结果一致性
    print("验证两种方法的一致性:")
    result1_method1 = transform_vectors(A1, B1, C1)
    result1_method2 = transform_vectors_numpy(A1, B1, C1)
    print(f"方法1: {result1_method1}")
    print(f"方法2: {result1_method2}")
    print(f"结果是否接近: {np.allclose(result1_method1, result1_method2, rtol=1e-10)}")