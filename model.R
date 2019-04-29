library(TreeBUGS)
head(arnold2013)

library(rstan)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

### Latent-Trait MPT model without parameter expansion, non-centered reparameterization

### model1(uni-demension)
m1 <- "
data { 
int<lower=1> nsubjs;  
int<lower=1> nparams; 
int<lower=0,upper=16> k_e[nsubjs,3];
int<lower=0,upper=16> k_u[nsubjs,3];
int<lower=0,upper=32> k_n[nsubjs,3];
}
parameters {
matrix[nparams,nsubjs] deltahat_tilde;

cholesky_factor_corr[nparams] L_Omega;
vector<lower=0>[nparams] sigma;

real mudhat;
real muchat;
real mughat;
real mubhat;
}
transformed parameters {
simplex[3] theta_e[nsubjs];
simplex[3] theta_u[nsubjs];
simplex[3] theta_n[nsubjs];
vector<lower=0,upper=1>[nsubjs] d;
vector<lower=0,upper=1>[nsubjs] c;
vector<lower=0,upper=1>[nsubjs] g;
vector<lower=0,upper=1>[nsubjs] b;

matrix[nsubjs,nparams] deltahat;

vector[nsubjs] deltadhat;
vector[nsubjs] deltachat;
vector[nsubjs] deltaghat;
vector[nsubjs] deltabhat;

deltahat = (diag_pre_multiply(sigma, L_Omega) * deltahat_tilde)'; 

for (i in 1:nsubjs) {

deltadhat[i] = deltahat[i,1];
deltachat[i] = deltahat[i,2];
deltaghat[i] = deltahat[i,3];
deltabhat[i] = deltahat[i,4];

// Probitize Parameters 
d[i] = Phi(mudhat + deltadhat[i]);
c[i] = Phi(muchat + deltachat[i]);
g[i] = Phi(mughat + deltaghat[i]);
b[i] = Phi(mubhat + deltabhat[i]);

// MPT Category Probabilities 
// EE
theta_e[i,1] = d[i]*c[i] + d[i]*(1-c[i])*g[i] + (1-d[i])*b[i]*g[i];
// EU
theta_e[i,2] = d[i]*(1-c[i])*(1-g[i]) + (1-d[i])*b[i]*(1-g[i]);
// EN
theta_e[i,3] = (1-d[i])*(1-b[i]);
// UE
theta_u[i,1] = d[i]*(1-c[i])*g[i] + (1-d[i])*b[i]*g[i];
// UU
theta_u[i,2] = d[i]*c[i] + d[i]*(1-c[i])*(1-g[i]) + (1-d[i])*b[i]*(1-g[i]);
// UN
theta_u[i,3] = (1-d[i])*(1-b[i]);
// NE
theta_n[i,1] = (1-d[i])*b[i]*g[i];
// NU
theta_n[i,2] = (1-d[i])*b[i]*(1-g[i]);
// NN
theta_n[i,3] =d[i] + (1-d[i])*(1-b[i]);
}
}
model {
// Priors
mudhat ~ normal(0, 1);
muchat ~ normal(0, 1);
mughat ~ normal(0, 1);
mubhat ~ normal(0, 1);

L_Omega ~ lkj_corr_cholesky(4); 
sigma ~ cauchy(0, 2.5); 
to_vector(deltahat_tilde) ~ normal(0, 1); 

// Data
for (i in 1:nsubjs) {
k_e[i] ~ multinomial(theta_e[i]);
k_u[i] ~ multinomial(theta_u[i]);
k_n[i] ~ multinomial(theta_n[i]);
}
}
generated quantities {
real<lower=0,upper=1> mud;
real<lower=0,upper=1> muc;
real<lower=0,upper=1> mug;
real<lower=0,upper=1> mub;
corr_matrix[nparams] Omega;
int<lower=0,upper=16> pred_e[nsubjs,3];
int<lower=0,upper=16> pred_u[nsubjs,3];
int<lower=0,upper=32> pred_n[nsubjs,3];
real log_lik[nsubjs];

// Post-Processing Means, Standard Deviations, Correlations
mud = Phi(mudhat);
muc = Phi(muchat);
mug = Phi(mughat);
mub = Phi(mubhat);

Omega = L_Omega * L_Omega';


// Predicted Data
for (i in 1:nsubjs) {
pred_e[i] = multinomial_rng(theta_e[i],16);
pred_u[i] = multinomial_rng(theta_u[i],16);
pred_n[i] = multinomial_rng(theta_n[i],32);
}

// log likelihood
for (i in 1:nsubjs)
log_lik[i] = multinomial_lpmf(k_e[i] | theta_e[i])+multinomial_lpmf(k_u[i] | theta_u[i])+multinomial_lpmf(k_n[i] | theta_n[i]);

}"

### model2(two demension)
m2 <- "
data { 
int<lower=1> nsubjs;  
int<lower=1> nparams; 
int<lower=0,upper=16> k_e[nsubjs,3];
int<lower=0,upper=16> k_u[nsubjs,3];
int<lower=0,upper=32> k_n[nsubjs,3];
}
parameters {
matrix[nparams,nsubjs] deltahat_tilde;

cholesky_factor_corr[nparams] L_Omega;
vector<lower=0>[nparams] sigma;

real mudhat;
real muchat;
real mughat;
real mubhat;
}
transformed parameters {
simplex[3] theta_e[nsubjs];
simplex[3] theta_u[nsubjs];
simplex[3] theta_n[nsubjs];
vector<lower=0,upper=1>[nsubjs] d;
vector<lower=0,upper=1>[nsubjs] c;
vector<lower=0,upper=1>[nsubjs] g;
vector<lower=0,upper=1>[nsubjs] b;

matrix[nsubjs,nparams] deltahat;

vector[nsubjs] deltadhat;
vector[nsubjs] deltachat;
vector[nsubjs] deltaghat;
vector[nsubjs] deltabhat;

deltahat = (diag_pre_multiply(sigma, L_Omega) * deltahat_tilde)'; 

for (i in 1:nsubjs) {

deltadhat[i] = deltahat[i,1];
deltachat[i] = deltahat[i,2];
deltaghat[i] = deltahat[i,3];
deltabhat[i] = deltahat[i,4];

// Probitize Parameters 
d[i] = Phi(mudhat + deltadhat[i]);
c[i] = Phi(muchat + deltachat[i]);
g[i] = Phi(mughat + deltaghat[i]);
b[i] = Phi(mubhat + deltabhat[i]);

// MPT Category Probabilities 
// EE
theta_e[i,1] = c[i]+(1-c[i])*d[i]*g[i]+(1-c[i])*(1-d[i])*b[i]*g[i];
// EU
theta_e[i,2] = (1-c[i])*d[i]*(1-g[i])+(1-c[i])*(1-d[i])*b[i]*(1-g[i]);
// EN
theta_e[i,3] = (1-c[i])*(1-d[i])*(1-b[i]);
// UE
theta_u[i,1] = (1-c[i])*d[i]*g[i]+(1-c[i])*(1-d[i])*b[i]*g[i];
// UU
theta_u[i,2] = c[i]+(1-c[i])*d[i]*(1-g[i])+(1-c[i])*(1-d[i])*b[i]*(1-g[i]);
// UN
theta_u[i,3] = (1-c[i])*(1-d[i])*(1-b[i]);
// NE
theta_n[i,1] = (1-d[i])*b[i]*g[i];
// NU
theta_n[i,2] = (1-d[i])*b[i]*(1-g[i]);
// NN
theta_n[i,3] =d[i] + (1-d[i])*(1-b[i]);
}
}
model {
// Priors
mudhat ~ normal(0, 1);
muchat ~ normal(0, 1);
mughat ~ normal(0, 1);
mubhat ~ normal(0, 1);

L_Omega ~ lkj_corr_cholesky(4); 
sigma ~ cauchy(0, 2.5); 
to_vector(deltahat_tilde) ~ normal(0, 1); 

// Data
for (i in 1:nsubjs) {
k_e[i] ~ multinomial(theta_e[i]);
k_u[i] ~ multinomial(theta_u[i]);
k_n[i] ~ multinomial(theta_n[i]);
}
}
generated quantities {
real<lower=0,upper=1> mud;
real<lower=0,upper=1> muc;
real<lower=0,upper=1> mug;
real<lower=0,upper=1> mub;
corr_matrix[nparams] Omega;
int<lower=0,upper=16> pred_e[nsubjs,3];
int<lower=0,upper=16> pred_u[nsubjs,3];
int<lower=0,upper=32> pred_n[nsubjs,3];
real log_lik[nsubjs];

// Post-Processing Means, Standard Deviations, Correlations
mud = Phi(mudhat);
muc = Phi(muchat);
mug = Phi(mughat);
mub = Phi(mubhat);

Omega = L_Omega * L_Omega';


// Predicted Data
for (i in 1:nsubjs) {
pred_e[i] = multinomial_rng(theta_e[i],16);
pred_u[i] = multinomial_rng(theta_u[i],16);
pred_n[i] = multinomial_rng(theta_n[i],32);
}

// log likelihood
for (i in 1:nsubjs)
log_lik[i] = multinomial_lpmf(k_e[i] | theta_e[i])+multinomial_lpmf(k_u[i] | theta_u[i])+multinomial_lpmf(k_n[i] | theta_n[i]);
}"





nparams = 4	 # number of parameters : 4
# 1. d = Di (detecting item probability)
# 2. c = Ds (detecting source probability)
# 3. g = Gs (guessing source probability - as expected)
# 4. b = Gi (guessing item probability - as old)		 

myinits <- list(
  list(deltahat=matrix(rnorm(24 * 4), 24, 4), Omega=diag(4),
       mudhat=rnorm(1),muchat=rnorm(1), mughat=rnorm(1), mubhat=rnorm(1),
       sigma=runif(4)),
  list(deltahat=matrix(rnorm(24 * 4), 24, 4), Omega=diag(4),
       mudhat=rnorm(1),muchat=rnorm(1), mughat=rnorm(1), mubhat=rnorm(1),
       sigma=runif(4)),
  list(deltahat=matrix(rnorm(24 * 4), 24, 4), Omega=diag(4),
       mudhat=rnorm(1),muchat=rnorm(1), mughat=rnorm(1), mubhat=rnorm(1),
       sigma=runif(4)),
  list(deltahat=matrix(rnorm(24 * 4), 24, 4), Omega=diag(4),
       mudhat=rnorm(1),muchat=rnorm(1), mughat=rnorm(1), mubhat=rnorm(1),
       sigma=runif(4)))


parameters <- c("mud","muc", "mug", "mub","Omega","d","c","g","b","pred_e","pred_u","pred_n","log_lik")  

myiterations <- 4000 
mywarmup <- 2000

enc_dat=arnold2013[1:24,] #data (encoding condition)
ret_dat=arnold2013[25:48,] #data (retrieval condition)

nsubjs <- 24	

enc_datalist <- list(k_e=enc_dat[,5:7],k_u=enc_dat[,8:10],k_n=enc_dat[,11:13], nparams=nparams, nsubjs=nsubjs)
ret_datalist <- list(k_e=ret_dat[,5:7],k_u=ret_dat[,8:10],k_n=ret_dat[,11:13], nparams=nparams, nsubjs=nsubjs)

# model1 / encoding condition
enc1 <- stan(model_code=m1, data=enc_datalist, 
            pars=parameters,iter=myiterations, warmup=mywarmup, init=myinits,
            chains=4, thin=1,control = list(adapt_delta = 0.9)
            )

# model1 / retrieval condition
ret1 <- stan(model_code=m1, data=ret_datalist, 
             pars=parameters,iter=myiterations, warmup=mywarmup, init=myinits,
             chains=4, thin=1,control = list(adapt_delta = 0.9999)
             )

# model2 / encoding condition
enc2 <- stan(model_code=m2, data=enc_datalist, 
             pars=parameters,iter=myiterations, warmup=mywarmup, init=myinits,
             chains=4, thin=1,control = list(adapt_delta = 0.9)
             )

# model2 / retrieval condition
ret2 <- stan(model_code=m2, data=ret_datalist, 
             pars=parameters,iter=myiterations, warmup=mywarmup, init=myinits,
             chains=4, thin=1,control = list(adapt_delta = 0.9)
             )


#looic

library(loo)

loo(extract_log_lik(enc1))
loo(extract_log_lik(enc2))
loo(extract_log_lik(ret1))
loo(extract_log_lik(ret2))

#posterior predictive plots

enc1_sub1=cbind(extract(enc1)$pred_e[,1,],extract(enc1)$pred_u[,1,],extract(enc1)$pred_n[,1,])
colnames(enc1_sub1)=c("EE","EU","EN","UE","UU","UN","NE","NU","NN")
boxplot(enc1_sub1)
points(as.numeric(enc_dat[1,5:13]),col="red",pch=17)

enc2_sub1=cbind(extract(enc2)$pred_e[,1,],extract(enc2)$pred_u[,1,],extract(enc2)$pred_n[,1,])
colnames(enc2_sub1)=c("EE","EU","EN","UE","UU","UN","NE","NU","NN")
boxplot(enc2_sub1)
points(as.numeric(enc_dat[1,5:13]),col="red",pch=17)

ret1_sub1=cbind(extract(ret1)$pred_e[,1,],extract(ret1)$pred_u[,1,],extract(ret1)$pred_n[,1,])
colnames(ret1_sub1)=c("EE","EU","EN","UE","UU","UN","NE","NU","NN",pch=17)
boxplot(ret1_sub1)
points(as.numeric(ret_dat[1,5:13]),col="red",pch=17)

ret2_sub1=cbind(extract(ret2)$pred_e[,1,],extract(ret2)$pred_u[,1,],extract(ret2)$pred_n[,1,])
colnames(ret2_sub1)=c("EE","EU","EN","UE","UU","UN","NE","NU","NN")
boxplot(ret2_sub1)
points(as.numeric(ret_dat[1,5:13]),col="red",pch=17)


# testing probability-matching theory

pc_r=arnold2013$pc[25:48]

gm_e1=apply(extract(enc1)$g, 2,mean )
gm_e2=apply(extract(enc2)$g,2,mean)
gm_r1=apply(extract(ret1)$g,2,mean)
gm_r2=apply(extract(ret2)$g,2,mean)

cm_e1=apply(extract(enc1)$c, 2,mean )
cm_e2=apply(extract(enc2)$c,2,mean)
cm_r1=apply(extract(ret1)$c,2,mean)
cm_r2=apply(extract(ret2)$c,2,mean)

cor(gm_e1,pc_e)
cor(gm_e2,pc_e)
cor(gm_r1,pc_r)
cor(gm_r2,pc_r)

plot(pc_e,gm_e1,pch=16,xlim=c(0,1),ylim=c(0,1),xlab="perceived contingency",ylab="source guessing probability:Gs",main="Encoding Condition/Model 1")
abline(a=0, b=1)

plot(pc_e,gm_e2,pch=16,xlim=c(0,1),ylim=c(0,1),xlab="perceived contingency",ylab="source guessing probability:Gs",main="Encoding Condition/Model 2")
plot(pc_r,gm_r1,pch=16,xlim=c(0,1),ylim=c(0,1),xlab="perceived contingency",ylab="source guessing probability:Gs",main="Retrieval Condition/Model 1")
plot(pc_r,gm_r2,pch=16,xlim=c(0,1),ylim=c(0,1),xlab="perceived contingency",ylab="source guessing probability:Gs",main="Retrieval Condition/Model 2")


plot(pcd[1:24],cm_e1,pch=16,xlim=c(0,0.3),ylim=c(0,1),xlab="deviation of perceived contingency from .5",ylab="source detecting probability:Ds",main="Encoding Condition/Model 1")
abline(lm(cm_e1~pcd[1:24]))

plot(pcd[1:24],cm_e2,pch=16,xlim=c(0,0.3),ylim=c(0,1),xlab="deviation of perceived contingency from .5",ylab="source detecting probability:Ds",main="Encoding Condition/Model 2")
abline(lm(cm_e2~pcd[1:24]))

plot(pcd[25:48],cm_r1,pch=16,xlim=c(0,0.5),ylim=c(0,1),xlab="deviation of perceived contingency from .5",ylab="source detecting probability:Ds",main="Retrieval Condition/Model 1")
abline(lm(cm_r1~pcd[25:48]))

plot(pcd[25:48],cm_r2,pch=16,xlim=c(0,0.5),ylim=c(0,1),xlab="deviation of perceived contingency from .5",ylab="source detecting probability:Ds",main="Retrieval Condition/Model 2")
abline(lm(cm_r2~pcd[25:48]))

cor(cm_e1,pcd[1:24])
cor(cm_e2,pcd[1:24])
cor(cm_r1,pcd[25:48])
cor(cm_r2,pcd[25:48])



pcd=abs(pc-0.5)



#estimates

d_e1 = extract(enc1)$mud
c_e1 = extract(enc1)$muc
g_e1 = extract(enc1)$mug
b_e1 = extract(enc1)$mub

d_e2 = extract(enc2)$mud
c_e2 = extract(enc2)$muc
g_e2 = extract(enc2)$mug
b_e2 = extract(enc2)$mub

d_r1 = extract(ret1)$mud
c_r1 = extract(ret1)$muc
g_r1 = extract(ret1)$mug
b_r1 = extract(ret1)$mub

d_r2 = extract(ret2)$mud
c_r2 = extract(ret2)$muc
g_r2 = extract(ret2)$mug
b_r2 = extract(ret2)$mub

dc_e1 <- extract(enc1)$Omega[, 1, 2]
dg_e1 <- extract(enc1)$Omega[, 1, 3]
db_e1 <- extract(enc1)$Omega[, 1, 4]
cg_e1 <- extract(enc1)$Omega[, 2, 3]
cb_e1 <- extract(enc1)$Omega[, 2, 4]
gb_e1 <- extract(enc1)$Omega[, 3, 4]

dc_e2 <- extract(enc2)$Omega[, 1, 2]
dg_e2 <- extract(enc2)$Omega[, 1, 3]
db_e2 <- extract(enc2)$Omega[, 1, 4]
cg_e2 <- extract(enc2)$Omega[, 2, 3]
cb_e2 <- extract(enc2)$Omega[, 2, 4]
gb_e2 <- extract(enc2)$Omega[, 3, 4]

dc_r1 <- extract(ret1)$Omega[, 1, 2]
dg_r1 <- extract(ret1)$Omega[, 1, 3]
db_r1 <- extract(ret1)$Omega[, 1, 4]
cg_r1 <- extract(ret1)$Omega[, 2, 3]
cb_r1 <- extract(ret1)$Omega[, 2, 4]
gb_r1 <- extract(ret1)$Omega[, 3, 4]

dc_r2 <- extract(ret2)$Omega[, 1, 2]
dg_r2 <- extract(ret2)$Omega[, 1, 3]
db_r2 <- extract(ret2)$Omega[, 1, 4]
cg_r2 <- extract(ret2)$Omega[, 2, 3]
cb_r2 <- extract(ret2)$Omega[, 2, 4]
gb_r2 <- extract(ret2)$Omega[, 3, 4]

# Plots posteriors for group mean parameters

plot(density(d_e1), xlim=c(0, 0.6), col="Firebrick", lty="dotted",lwd=1.5,
     ylab="Probability Density", xlab=expression(mu[Di]), main="", 
     yaxt="n", xaxt="n")
lines(density(d_e2),col="Firebrick",lty="solid",lwd=1.5)
lines(density(d_r1),col="Midnight Blue",lty="dotted",lwd=1.5)
lines(density(d_r2), col="Midnight Blue",lty="solid",lwd=1.5)
legend("topright",legend=c("enc/m1","enc/m2","ret/m1","ret/m2")
       ,col=c("Firebrick","Firebrick","Midnight Blue","Midnight Blue"),
       lty=c("dotted","solid","dotted","solid"))
axis(1, seq(0, 0.6, by=.2), tick=T)



plot(density(c_e1), xlim=c(0,1), ylim=c(0,17), col="Firebrick", lty="dotted",lwd=1.5,
     ylab="Probability Density", xlab=expression(mu[Ds]), main="", 
     yaxt="n", xaxt="n")
lines(density(c_e2),col="Firebrick",lty="solid",lwd=1.5)
lines(density(c_r1),col="Midnight Blue",lty="dotted",lwd=1.5)
lines(density(c_r2), col="Midnight Blue",lty="solid",lwd=1.5)
legend("topright",legend=c("enc/m1","enc/m2","ret/m1","ret/m2")
       ,col=c("Firebrick","Firebrick","Midnight Blue","Midnight Blue"),
       lty=c("dotted","solid","dotted","solid"))
axis(1, seq(0, 1, by=.2), tick=T)

plot(density(g_e1), xlim=c(0.2,1), ylim=c(0,8), col="Firebrick", lty="dotted",lwd=1.5,
     ylab="Probability Density", xlab=expression(mu[Gs]), main="", 
     yaxt="n", xaxt="n")
lines(density(g_e2),col="Firebrick",lty="solid",lwd=1.5)
lines(density(g_r1),col="Midnight Blue",lty="dotted",lwd=1.5)
lines(density(g_r2), col="Midnight Blue",lty="solid",lwd=1.5)
legend("topleft",legend=c("enc/m1","enc/m2","ret/m1","ret/m2")
       ,col=c("Firebrick","Firebrick","Midnight Blue","Midnight Blue"),
       lty=c("dotted","solid","dotted","solid"))
axis(1, seq(0.2, 1, by=.2), tick=T)

plot(density(b_e1), xlim=c(0.2,0.8), ylim=c(0,11), col="Firebrick", lty="dotted",lwd=1.5,
     ylab="Probability Density", xlab=expression(mu[Gi]), main="", 
     yaxt="n", xaxt="n")
lines(density(b_e2),col="Firebrick",lty="solid",lwd=1.5)
lines(density(b_r1),col="Midnight Blue",lty="dotted",lwd=1.5)
lines(density(b_r2), col="Midnight Blue",lty="solid",lwd=1.5)
legend("topright",legend=c("enc/m1","enc/m2","ret/m1","ret/m2")
       ,col=c("Firebrick","Firebrick","Midnight Blue","Midnight Blue"),
       lty=c("dotted","solid","dotted","solid"))
axis(1, seq(0.2, 1, by=.2), tick=T)



# Plots posteriors for the correlations
plot(density(dc_e1), xlim=c(-1,1), ylim=c(0,2), col="Firebrick", lty="dotted",lwd=1.5,
     ylab="Probability Density", xlab=expression(rho[DiDs]), main="", 
     yaxt="n", xaxt="n")
lines(density(dc_e2),col="Firebrick",lty="solid",lwd=1.5)
lines(density(dc_r1),col="Midnight Blue",lty="dotted",lwd=1.5)
lines(density(dc_r2), col="Midnight Blue",lty="solid",lwd=1.5)
legend("topleft",legend=c("enc/m1","enc/m2","ret/m1","ret/m2")
       ,col=c("Firebrick","Firebrick","Midnight Blue","Midnight Blue"),
       lty=c("dotted","solid","dotted","solid"))
axis(1, seq(-1, 1, by=.2), tick=T)

plot(density(dg_e1), xlim=c(-1,1), ylim=c(0,2), col="Firebrick", lty="dotted",lwd=1.5,
     ylab="Probability Density", xlab=expression(rho[DiGs]), main="", 
     yaxt="n", xaxt="n")
lines(density(dg_e2),col="Firebrick",lty="solid",lwd=1.5)
lines(density(dg_r1),col="Midnight Blue",lty="dotted",lwd=1.5)
lines(density(dg_r2), col="Midnight Blue",lty="solid",lwd=1.5)
legend("topleft",legend=c("enc/m1","enc/m2","ret/m1","ret/m2")
       ,col=c("Firebrick","Firebrick","Midnight Blue","Midnight Blue"),
       lty=c("dotted","solid","dotted","solid"))
axis(1, seq(-1, 1, by=.2), tick=T)

plot(density(db_e1), xlim=c(-1,1), ylim=c(0,2), col="Firebrick", lty="dotted",lwd=1.5,
     ylab="Probability Density", xlab=expression(rho[DiGi]), main="", 
     yaxt="n", xaxt="n")
lines(density(db_e2),col="Firebrick",lty="solid",lwd=1.5)
lines(density(db_r1),col="Midnight Blue",lty="dotted",lwd=1.5)
lines(density(db_r2), col="Midnight Blue",lty="solid",lwd=1.5)
legend("topleft",legend=c("enc/m1","enc/m2","ret/m1","ret/m2")
       ,col=c("Firebrick","Firebrick","Midnight Blue","Midnight Blue"),
       lty=c("dotted","solid","dotted","solid"))
axis(1, seq(-1, 1, by=.2), tick=T)


plot(density(cg_e1), xlim=c(-1,1), ylim=c(0,2.5), col="Firebrick", lty="dotted",lwd=1.5,
     ylab="Probability Density", xlab=expression(rho[DsGs]), main="", 
     yaxt="n", xaxt="n")
lines(density(cg_e2),col="Firebrick",lty="solid",lwd=1.5)
lines(density(cg_r1),col="Midnight Blue",lty="dotted",lwd=1.5)
lines(density(cg_r2), col="Midnight Blue",lty="solid",lwd=1.5)
legend("topright",legend=c("enc/m1","enc/m2","ret/m1","ret/m2")
       ,col=c("Firebrick","Firebrick","Midnight Blue","Midnight Blue"),
       lty=c("dotted","solid","dotted","solid"))
axis(1, seq(-1, 1, by=.2), tick=T)

plot(density(cb_e1), xlim=c(-1,1), ylim=c(0,2), col="Firebrick", lty="dotted",lwd=1.5,
     ylab="Probability Density", xlab=expression(rho[DsGi]), main="", 
     yaxt="n", xaxt="n")
lines(density(cb_e2),col="Firebrick",lty="solid",lwd=1.5)
lines(density(cb_r1),col="Midnight Blue",lty="dotted",lwd=1.5)
lines(density(cb_r2), col="Midnight Blue",lty="solid",lwd=1.5)
legend("topright",legend=c("enc/m1","enc/m2","ret/m1","ret/m2")
       ,col=c("Firebrick","Firebrick","Midnight Blue","Midnight Blue"),
       lty=c("dotted","solid","dotted","solid"))
axis(1, seq(-1, 1, by=.2), tick=T)

plot(density(gb_e1), xlim=c(-1,1), ylim=c(0,2), col="Firebrick", lty="dotted",lwd=1.5,
     ylab="Probability Density", xlab=expression(rho[GsGi]), main="", 
     yaxt="n", xaxt="n")
lines(density(gb_e2),col="Firebrick",lty="solid",lwd=1.5)
lines(density(gb_r1),col="Midnight Blue",lty="dotted",lwd=1.5)
lines(density(gb_r2), col="Midnight Blue",lty="solid",lwd=1.5)
legend("topright",legend=c("enc/m1","enc/m2","ret/m1","ret/m2")
       ,col=c("Firebrick","Firebrick","Midnight Blue","Midnight Blue"),
       lty=c("dotted","solid","dotted","solid"))
axis(1, seq(-1, 1, by=.2), tick=T)
