# ------------------------------------------------------------
# 1. 시뮬레이션 환경 및 데이터 생성
# ------------------------------------------------------------
set.seed(2026) # 재현성을 위한 시드 고정
N <- 1500

# 가상의 독립변수 생성
x1 <- rnorm(N)
x2 <- rnorm(N)

# Design Matrix 구성 (절편 포함)
# X: Count part (Poisson) 독립변수
# Z: Inflation part (Zero) 독립변수 - X와 다르게 구성해봄
X <- cbind(1, x1, x2) 
Z <- cbind(1, x1)     

# 실제 모수(True Parameters) 설정
true_beta <- c(1.2, 0.4, -0.3)  # Count part 모수
true_gamma <- c(-0.8, 1.0)      # Inflation part 모수

# 선형 예측자를 통한 파라미터 계산
mu_true <- exp(X %*% true_beta)
pi_true <- 1 / (1 + exp(-(Z %*% true_gamma)))

# 난수 생성을 통한 Y 값 도출
# 1. 해당 관측치가 구조적 0(Inflation part)에 속하는지 여부 (1 = 구조적 0)
is_structural_zero <- rbinom(N, size = 1, prob = pi_true)

# 2. Count part에서의 Poisson 발생 건수
poisson_counts <- rpois(N, lambda = mu_true)

# 3. 최종 관측 데이터 Y
# 구조적 0 그룹이면 무조건 0, 아니면 Poisson 분포에서 나온 값
Y <- ifelse(is_structural_zero == 1, 0, poisson_counts)


# ------------------------------------------------------------
# 2. 모델 적합 (작성한 zip_em_irls 함수 테스트)
# ------------------------------------------------------------
cat("Fitting ZIP model using Custom EM-IRLS algorithm...\n")
res <- zip_em_irls(X = X, Z = Z, Y = Y, max_iter = 500, tol = 1e-6)


# ------------------------------------------------------------
# 3. 결과 비교 출력
# ------------------------------------------------------------
cat("\n=================================================\n")
cat("               [ 파라미터 추정 결과 ]\n")
cat("=================================================\n")

cat("\n1. Count Part (Beta 계수)\n")
cat(" - True Beta : ", sprintf("%6.3f", true_beta), "\n")
cat(" - Est  Beta : ", sprintf("%6.3f", res$beta), "\n")

cat("\n2. Inflation Part (Gamma 계수)\n")
cat(" - True Gamma: ", sprintf("%6.3f", true_gamma), "\n")
cat(" - Est  Gamma: ", sprintf("%6.3f", res$gamma), "\n")

cat("\n3. EM Algorithm 정보\n")
cat(" - 반복 횟수(Iterations) : ", res$em_iterations, "\n")
cat(" - 최종 Q-value (LogLik) : ", round(res$Q_value, 4), "\n")
cat("=================================================\n")


# ------------------------------------------------------------
# 4. 수렴 과정 시각화 (Plotting)
# ------------------------------------------------------------
# 이전에 작성하신 plot_cwm 스타일처럼 Q-value의 변화를 그래프로 그립니다.
plot(res$Q_history, type = "b", pch = 19, col = "royalblue",
     main = "EM Algorithm Convergence (Q-value)",
     xlab = "EM Iteration", ylab = "Q-value (Expected Log-Likelihood)",
     lwd = 2)
grid()