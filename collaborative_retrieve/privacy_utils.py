import numpy as np
import base64
import os
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding as sym_padding

class PrivacyUtils:
    @staticmethod
    def add_noise(vector, noise_level=0.01):
        """向向量添加高斯噪声 (Differential Privacy 思想)"""
        vec = np.array(vector)
        norm = np.linalg.norm(vec) + 1e-9
        vec_normalized = vec / norm
        
        # 生成噪声
        noise = np.random.normal(0, noise_level, vec.shape)
        noisy_vec = vec_normalized + noise
        
        # 重新归一化
        return (noisy_vec / np.linalg.norm(noisy_vec)).tolist()

    @staticmethod
    def generate_keys():
        """生成 RSA 密钥对"""
        priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pub = priv.public_key()
        pub_pem = pub.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        return priv, pub_pem

    @staticmethod
    def encrypt(data_str, pub_key_pem):
        """混合加密：RSA 加密 AES Key，AES 加密数据 (模拟不经意传输)"""
        # 1. 准备 AES 密钥
        aes_key = os.urandom(32)
        iv = os.urandom(16)
        
        # 2. AES 加密内容
        padder = sym_padding.PKCS7(128).padder()
        padded_data = padder.update(data_str.encode()) + padder.finalize()
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        enc_content = encryptor.update(padded_data) + encryptor.finalize()
        
        # 3. RSA 加密 AES 密钥
        pub_key = serialization.load_pem_public_key(pub_key_pem)
        enc_key = pub_key.encrypt(
            aes_key,
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
        )
        
        return {
            "iv": base64.b64encode(iv).decode(),
            "k": base64.b64encode(enc_key).decode(),
            "d": base64.b64encode(enc_content).decode()
        }

    @staticmethod
    def decrypt(enc_pkg, priv_key):
        """解密数据"""
        iv = base64.b64decode(enc_pkg['iv'])
        enc_key = base64.b64decode(enc_pkg['k'])
        enc_data = base64.b64decode(enc_pkg['d'])
        
        # 1. RSA 解密 AES Key
        aes_key = priv_key.decrypt(
            enc_key,
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
        )
        
        # 2. AES 解密内容
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(enc_data) + decryptor.finalize()
        unpadder = sym_padding.PKCS7(128).unpadder()
        return (unpadder.update(padded_data) + unpadder.finalize()).decode()