"""
pip install seal

This example demonstrates how to use Microsoft SEAL
to perform privacy-preserving sentiment analysis on communication data.
"""

import seal
from seal import EncryptionParameters, SEALContext, KeyGenerator, IntegerEncoder, Encryptor, Decryptor, Evaluator, scheme_type, Plaintext, Ciphertext
import math

# Step 1: Set encryption parameters (BFV scheme)
parms = EncryptionParameters(scheme_type.BFV)
poly_modulus_degree = 4096
parms.set_poly_modulus_degree(poly_modulus_degree)
parms.set_coeff_modulus(seal.CoeffModulus.BFVDefault(poly_modulus_degree))
parms.set_plain_modulus(1031)  # prime number > max plaintext

# Step 2: Create SEALContext
context = SEALContext.Create(parms)

# Step 3: Key generation
keygen = KeyGenerator(context)
public_key = keygen.public_key()
secret_key = keygen.secret_key()

# Step 4: Setup tools
encryptor = Encryptor(context, public_key)
decryptor = Decryptor(context, secret_key)
evaluator = Evaluator(context)
encoder = IntegerEncoder(context)

# Step 5: Simulated sentiment scores (output from NLP model)
# These would come from analysis of chat/email/transcript
sentiment_scores = [7, 8, 6]  # sample scores out of 10

# Step 6: Encrypt each score
encrypted_scores = []
for score in sentiment_scores:
    plain = encoder.encode(score)
    encrypted = encryptor.encrypt(plain)
    encrypted_scores.append(encrypted)

# Step 7: Compute encrypted sum
encrypted_sum = encrypted_scores[0]
for i in range(1, len(encrypted_scores)):
    evaluator.add_inplace(encrypted_sum, encrypted_scores[i])

# Step 8: Compute average (divide by count)
# Use plain encoding of reciprocal (1/3 ≈ 0.33)
# Since IntegerEncoder doesn’t support fractions, simulate via scaled integer
# Multiply by 100 and divide later
scale = 100
inverse = math.floor(scale / len(sentiment_scores))  # 33

plain_inverse = encoder.encode(inverse)
evaluator.multiply_plain_inplace(encrypted_sum, plain_inverse)

# Step 9: Decrypt and decode
decrypted_result = decryptor.decrypt(encrypted_sum)
decoded_result = encoder.decode_int32(decrypted_result)

# Step 10: Final result after scaling back
average_score = decoded_result / scale

print(f"Encrypted sentiment scores: {sentiment_scores}")
print(f"Privacy-preserving average (decrypted): {average_score}")
