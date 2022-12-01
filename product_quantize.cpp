#include<bits/stdc++.h>
using namespace std;

template <typename A, typename B>
string to_string(pair<A, B> p);
 
template <typename A, typename B, typename C>
string to_string(tuple<A, B, C> p);
 
template <typename A, typename B, typename C, typename D>
string to_string(tuple<A, B, C, D> p);

string to_string(const string& s) {
    return '"' + s + '"';
}

string to_string(const char& s) {
    return to_string(string(1, s));
}
 
string to_string(const char* s) {
    return to_string((string) s);
}
 
string to_string(const bool& b) {
    return (b ? "1" : "0");
}
 
string to_string(const vector<bool>& v) {
	bool first = true;
	string res = "{";
	for (int i = 0; i < static_cast<int>(v.size()); i++) {
		if (!first) {
			res += ", ";
		}
		first = false;
		res += to_string(v[i]);
	}
	res += "}";
	return res;
}
 
template <size_t N>
string to_string(const bitset<N>& v) {
	string res = "";
	for (size_t i = 0; i < N; i++) {
		res += static_cast<char>('0' + v[i]);
	}
	return res;
}
 
template <typename A>
string to_string(const A& v) {
	bool first = true;
	string res = "{";
	for (const auto &x : v) {
		if (!first) {
			res += ", ";
		}
		first = false;
		res += to_string(x);
	}
	res += "}";
	return res;
}
 
template <typename A, typename B>
string to_string(pair<A, B> p) {
	return "(" + to_string(p.first) + ", " + to_string(p.second) + ")";
}
 
template <typename A, typename B, typename C>
string to_string(tuple<A, B, C> p) {
	return "(" + to_string(get<0>(p)) + ", " + to_string(get<1>(p)) + ", " + to_string(get<2>(p)) + ")";
}
 
template <typename A, typename B, typename C, typename D>
string to_string(tuple<A, B, C, D> p) {
    return "(" + to_string(get<0>(p)) + ", " + to_string(get<1>(p)) + ", " + to_string(get<2>(p)) + ", " + to_string(get<3>(p)) + ")";
}
 
void debug_out() { cerr << endl; }
 
template <typename Head, typename... Tail>
void debug_out(Head H, Tail... T) {
    cerr << " " << to_string(H);
    debug_out(T...);
}
 
#ifdef XOX // use -DXOX flag while compiling
#define debug(...) cerr << "[" << #__VA_ARGS__ << "]:", debug_out(__VA_ARGS__)
#else
#define debug(...) 42
#endif

std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count()); // use mt19937_64 for 64 bit
const float epsilon = 1e-3;

namespace error {

float rmse(const vector<float>& v1, const vector<float>& v2) {
    assert(v1.size() == v2.size());
    float ans = 0;
    for(size_t i = 0; i < v1.size(); i++) {
        float diff = v1[i] - v2[i];
        ans += diff * diff;
    }
    return sqrtf(ans);
}
}

float distL2(const vector<float>& v1, const vector<float>& v2) {
    // Euclidean distance
    assert(v1.size() == v2.size());
    float ans = 0;
    for(size_t i = 0; i < v1.size(); i++) {
        float diff = v1[i] - v2[i];
        ans += diff * diff;
    }
    return sqrtf(ans);
}

void add_vector_self(vector<float>& v1, const vector<float>& v2) {
    assert(v1.size() == v2.size());
    for(size_t i = 0; i < v1.size(); i++) {
        v1[i] += v2[i];
    }
}

void divide_vector_self(vector<float>& v1, const int n) {
    for(size_t i = 0; i < v1.size(); i++) {
        v1[i] /= (float)n;
    }
}

class ProductQuantize {
    public:
    std::vector<std::vector<float>> X; // Database
    std::vector<std::vector<float>> X_compressed; // Compressed database
    std::vector<std::vector<std::vector<float>>> X_partitioned; // Partitioned database
    std::vector<std::vector<std::vector<float>>> codebook;
    int N; // Number of vectors in X
    int L; // Length of each vector in X
    int M; // Number of partitions
    int K; // Number of centroids
    int L_centroid; // Length of each centroid = L / M

    ProductQuantize() {}
    ProductQuantize(std::vector<std::vector<float>>& _X, int _M = 8, int _K = 256) {
        X = _X, N = X.size(), L = X[0].size(), M = _M, K = _K;
        assert(L % M == 0);
        L_centroid = L / M;
        codebook.resize(K, std::vector<std::vector<float>>(M, std::vector<float>(L_centroid)));
        X_partitioned.resize(N, std::vector<std::vector<float>>(M, std::vector<float>(L_centroid)));
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < L; j++) {
                X_partitioned[i][j / L_centroid][j % L_centroid] = X[i][j];
            }
        }
        generate_codebook();
        // Compress the database X
        X_compressed.resize(N, vector<float>(M));
        for(int i = 0; i < N; i++) {
            for(int m = 0; m < M; m++) {
                int i_k = -1;
                float min_dis = 1e9;
                for(int k = 0; k < K; k++) {
                    float cur_dis = distL2(codebook[k][m], X_partitioned[i][m]);
                    if(cur_dis < min_dis) {
                        min_dis = cur_dis;
                        i_k = k;
                    }
                }
                X_compressed[i][m] = i_k;
            }
        }
    }

    void initialize_codebook() {
        // Pick K random vectors from the database as the initial centroids
        vector<int> perm(N);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        for(int i = 0; i < K; i++) {
            for(int j = 0; j < L; j++) {
                codebook[i][j / L_centroid][j % L_centroid] = X[perm[i]][j];
            }
        }
    }

    void k_means(int num_iter = 50) {
        while(num_iter > 0) {
            // Assign labels (nearest centroid) to all vectors
            vector<vector<int>> label(N, vector<int>(M, -1));
            for(int i = 0; i < N; i++) {
                for(int m = 0; m < M; m++) {
                    float min_dis = 1e9;
                    for(int k = 0; k < K; k++) {
                        float cur_dis = distL2(codebook[k][m], X_partitioned[i][m]);
                        if(cur_dis < min_dis) {
                            min_dis = cur_dis;
                            label[i][m] = k;
                        }
                    }
                }
            }
            // Update centroid to mean of all vectors
            vector<vector<vector<float>>> ncodebook(K, std::vector<std::vector<float>>(M, std::vector<float>(L_centroid)));
            vector<vector<int>> cnt(K, vector<int>(M));
            for(int i = 0; i < N; i++) {
                for(int m = 0; m < M; m++) {
                    int k = label[i][m];
                    cnt[k][m]++;
                    add_vector_self(ncodebook[k][m], X_partitioned[i][m]);
                }
            }
            for(int k = 0; k < K; k++) {
                for(int m = 0; m < M; m++) {
                    divide_vector_self(ncodebook[k][m], cnt[k][m]);
                }
            }
            // Check the diff btw new and old codebook
            float max_change = 0;
            for(int k = 0; k < K; k++) {
                for(int m = 0; m < M; m++) {
                    float cur_change = distL2(ncodebook[k][m], codebook[k][m]);
                    max_change = max(max_change, cur_change);
                }
            }
            swap(codebook, ncodebook);
            if(max_change < epsilon) {
                break;
            }
            num_iter--;
        }
    }

    void generate_codebook() {
        initialize_codebook();
        k_means();
    }

    vector<float> query_approx(vector<float> q) {
        assert((int)q.size() == L);
        vector<vector<float>> q_partioned(M, vector<float>(L / M));
        for(int i = 0; i < L; i++) {
            q_partioned[i / L_centroid][i % L_centroid] = q[i];
        }
        // Compute LUT (Look Up Table)
        vector<vector<float>> lut(K, vector<float>(M));
        for(int m = 0; m < M; m++) {
            for(int k = 0; k < K; k++) {
                lut[k][m] = distL2(q_partioned[m], codebook[k][m]);
            }
        }
        // Compute dist to all vectors in database
        vector<float> res(N);
        for(int i = 0; i < N; i++) {
            for(int m = 0; m < M; m++) {
                int i_k = X_compressed[i][m];
                res[i] += lut[i_k][m];
            }
        }
        return res;
    }

    vector<float> query_exact(vector<float> q) {
        assert((int)q.size() == L);
        vector<float> res(N);
        for(int i = 0; i < N; i++) {
            res[i] = distL2(X[i], q);
        }
        return res;
    }
};

int main() {
    // fast io
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    // code starts here
    
    // Generate random vectors for database
    int N = 10000;
    int L = 256;
    vector<vector<float>> X(N, vector<float>(L));
    std::uniform_real_distribution<float> urd(0, 10);
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < L; j++) {
            X[i][j] = urd(rng);
        }
    }
    float tb = clock() * 1.0 / CLOCKS_PER_SEC;
    ProductQuantize pq(X, 8, 512);
    float ta = clock() * 1.0 / CLOCKS_PER_SEC;
    debug(ta - tb);
    vector<float> q(L);
    for(int i = 0; i < L; i++) {
        q[i] = urd(rng);
    }
    float t1 = clock() * 1.0 / CLOCKS_PER_SEC;
    auto res_exact = pq.query_exact(q);
    float t2 = clock() * 1.0 / CLOCKS_PER_SEC;
    debug(t2 - t1);
    float t3 = clock() * 1.0 / CLOCKS_PER_SEC;
    auto res_approx = pq.query_approx(q);
    float t4 = clock() * 1.0 / CLOCKS_PER_SEC;
    debug(t4 - t3);
    float err = error::rmse(res_exact, res_approx);
    debug(err);
    return 0;
}