#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <random>
#include <iomanip>
#include <string>

static inline uint32_t rotateleft32(uint32_t x, unsigned int n) {
    n &= 31;
    return (x << n) | (x >> (32 - n));
}

static inline uint32_t rotateright32(uint32_t x, unsigned int n) {
    n &= 31;
    return (x >> n) | (x << (32 - n));
}

static inline uint32_t tensorProduct32(uint32_t K, uint32_t L) {
    uint8_t K_bytes[4] = {
            static_cast<uint8_t>((K >> 24) & 0xFF),
            static_cast<uint8_t>((K >> 16) & 0xFF),
            static_cast<uint8_t>((K >> 8) & 0xFF),
            static_cast<uint8_t>(K & 0xFF)
    };

    uint8_t L_bytes[4] = {
            static_cast<uint8_t>((L >> 24) & 0xFF),
            static_cast<uint8_t>((L >> 16) & 0xFF),
            static_cast<uint8_t>((L >> 8) & 0xFF),
            static_cast<uint8_t>(L & 0xFF)
    };

    uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result ^= K_bytes[i] * L_bytes[j];
        }
    }

    return result & 0xFFFFFFFF;
}


static inline uint64_t rotateright64(uint64_t x, unsigned int n) {
    n &= 63;
    return (x >> n) | (x << (64 - n));
}

uint32_t F_func(uint32_t L, uint32_t K) {
    uint32_t leftShifted = rotateleft32(L, 9);

    uint32_t tensorProd = tensorProduct32(rotateright32(K, 11), L);

    return leftShifted ^ (~tensorProd);
}

void feistelRound(uint32_t &L, uint32_t &R, uint32_t Ki) {
    uint32_t newL = R ^ F_func(L, Ki);
    uint32_t newR = L;
    L = newL;
    R = newR;
}

void feistelRoundInverse(uint32_t &L, uint32_t &R, uint32_t Ki) {
    uint32_t prevL = R;
    uint32_t prevR = L ^ F_func(R, Ki);
    L = prevL;
    R = prevR;
}


std::vector<uint32_t> generateRoundKeys(uint64_t K, int rounds) {
    std::vector<uint32_t> keys;
    keys.reserve(rounds);
    for (int i = 0; i < rounds; ++i) {
        uint64_t t = rotateright64(K, static_cast<unsigned int>(i * 3));
        uint32_t Ki = static_cast<uint32_t>(t & 0xFFFFFFFFu);
        keys.push_back(Ki);
    }
    return keys;
}

uint64_t feistelEncryptBlock(uint64_t block, const std::vector<uint32_t>& keys) {
    uint32_t L = static_cast<uint32_t>(block >> 32);
    uint32_t R = static_cast<uint32_t>(block & 0xFFFFFFFFu);
    for (auto Ki : keys)
        feistelRound(L, R, Ki);
    return (static_cast<uint64_t>(L) << 32) | static_cast<uint64_t>(R);
}

uint64_t feistelDecryptBlock(uint64_t block, const std::vector<uint32_t>& keys) {
    uint32_t L = static_cast<uint32_t>(block >> 32);
    uint32_t R = static_cast<uint32_t>(block & 0xFFFFFFFFu);
    for (int i = static_cast<int>(keys.size()) - 1; i >= 0; --i)
        feistelRoundInverse(L, R, keys[i]);
    return (static_cast<uint64_t>(L) << 32) | static_cast<uint64_t>(R);
}

bool encryptFile(const std::string &inPath, const std::string &outPath, uint64_t K, int rounds) {
    if (rounds < 8) return false;
    auto keys = generateRoundKeys(K, rounds);
    std::ifstream fin(inPath, std::ios::binary);
    if (!fin) return false;
    std::ofstream fout(outPath, std::ios::binary);
    if (!fout) return false;
    fin.seekg(0, std::ios::end);
    uint64_t filesize = static_cast<uint64_t>(fin.tellg());
    fin.seekg(0, std::ios::beg);
    fout.write(reinterpret_cast<const char*>(&filesize), sizeof(filesize));
    const size_t blockBytes = 8;
    std::vector<char> buf(blockBytes, 0);
    while (true) {
        fin.read(buf.data(), blockBytes);
        std::streamsize readn = fin.gcount();
        if (readn <= 0) break;
        uint64_t block = 0;
        memcpy(&block, buf.data(), static_cast<size_t>(readn));
        uint64_t cipher = feistelEncryptBlock(block, keys);
        fout.write(reinterpret_cast<const char*>(&cipher), sizeof(cipher));
        if (readn < static_cast<std::streamsize>(blockBytes)) break;
    }
    return true;
}

bool decryptFile(const std::string &inPath, const std::string &outPath, uint64_t K, int rounds) {
    if (rounds < 8) return false;
    auto keys = generateRoundKeys(K, rounds);
    std::ifstream fin(inPath, std::ios::binary);
    if (!fin) return false;
    uint64_t orig_size = 0;
    fin.read(reinterpret_cast<char*>(&orig_size), sizeof(orig_size));
    if (!fin) return false;
    std::ofstream fout(outPath, std::ios::binary);
    if (!fout) return false;
    const size_t blockBytes = 8;
    uint64_t total_written = 0;
    while (true) {
        uint64_t cipher = 0;
        fin.read(reinterpret_cast<char*>(&cipher), sizeof(cipher));
        if (fin.gcount() <= 0) break;
        uint64_t plain = feistelDecryptBlock(cipher, keys);
        size_t to_write = blockBytes;
        if (total_written + to_write > orig_size)
            to_write = static_cast<size_t>(orig_size - total_written);
        fout.write(reinterpret_cast<const char*>(&plain), to_write);
        total_written += to_write;
        if (total_written >= orig_size) break;
    }
    return true;
}

int main() {
    std::cout << "Choose: 1 - Encrypt, 2 - Decrypt: ";
    int mode;
    if (!(std::cin >> mode)) return 1;

    std::string inPath, outPath;
    std::cout << "Input file: ";
    std::cin >> inPath;
    std::cout << "Output file: ";
    std::cin >> outPath;
    int rounds;
    std::cout << "Enter count of rounds: ";
    std::cin >> rounds;
    if (rounds < 8) return 1;

    uint64_t K = 0;
    bool ok = false;

    if (mode == 1) {
        std::random_device seed;
        std::mt19937_64 gen(seed());
        std::uniform_int_distribution<uint64_t> dist(0);
        K = dist(gen);

        std::ofstream keyOut("key.txt");
        keyOut << std::hex << std::setw(16) << std::setfill('0') << K;
        keyOut.close();

        std::cout << "Generated key saved to key.txt: 0x"
                  << std::hex << std::setw(16) << std::setfill('0') << K << std::dec << "\n";

        std::cout << "Encrypting...\n";
        ok = encryptFile(inPath, outPath, K, rounds);
    }
    else if (mode == 2) {
        std::ifstream keyIn("key.txt");
        if (!keyIn) {
            std::cerr << "Error: key.txt not found!\n";
            return 1;
        }
        keyIn >> std::hex >> K;
        keyIn.close();

        std::cout << "Loaded key: 0x"
                  << std::hex << std::setw(16) << std::setfill('0') << K << std::dec << "\n";

        std::cout << "Decrypting...\n";
        ok = decryptFile(inPath, outPath, K, rounds);
    }
    else return 1;

    if (ok) std::cout << "Done.\n";
    else std::cout << "Error.\n";
    return ok ? 0 : 1;
}
