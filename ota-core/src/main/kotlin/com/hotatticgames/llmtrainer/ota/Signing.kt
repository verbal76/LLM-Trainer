package com.hotatticgames.llmtrainer.ota

import java.security.KeyFactory
import java.security.KeyPairGenerator
import java.security.PrivateKey
import java.security.PublicKey
import java.security.Signature
import java.security.spec.ECGenParameterSpec
import java.security.spec.PKCS8EncodedKeySpec
import java.security.spec.X509EncodedKeySpec
import java.util.Base64

/** A trusted verification key. Several can be trusted at once to allow key rotation. */
data class TrustedKey(val keyId: String, val publicKeyDerBase64: String)

/** ECDSA P-256 / SHA-256: available on every supported Android API level and on the JVM. */
object Signing {
    private const val ALG = "SHA256withECDSA"

    class Generated(val publicDerBase64: String, val privateDerBase64: String)

    fun generate(): Generated {
        val g = KeyPairGenerator.getInstance("EC")
        g.initialize(ECGenParameterSpec("secp256r1"))
        val kp = g.generateKeyPair()
        val enc = Base64.getEncoder()
        return Generated(enc.encodeToString(kp.public.encoded), enc.encodeToString(kp.private.encoded))
    }

    fun sign(data: ByteArray, privateDerBase64: String): String {
        val pk = privateKey(privateDerBase64)
        val s = Signature.getInstance(ALG)
        s.initSign(pk)
        s.update(data)
        return Base64.getEncoder().encodeToString(s.sign())
    }

    fun verify(data: ByteArray, sigBase64: String, key: TrustedKey): Boolean = try {
        val s = Signature.getInstance(ALG)
        s.initVerify(publicKey(key.publicKeyDerBase64))
        s.update(data)
        s.verify(Base64.getDecoder().decode(sigBase64.trim()))
    } catch (_: Exception) {
        false
    }

    private fun publicKey(b64: String): PublicKey =
        KeyFactory.getInstance("EC").generatePublic(X509EncodedKeySpec(Base64.getDecoder().decode(b64.trim())))

    private fun privateKey(b64: String): PrivateKey =
        KeyFactory.getInstance("EC").generatePrivate(PKCS8EncodedKeySpec(Base64.getDecoder().decode(b64.trim())))
}
