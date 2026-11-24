# How the networks forecast a wind map

Let $X_t$ be the 11-field weather map at time $t$, standardized field by field using its training window. Each map has 5 latitude rows and 7 longitude columns. At forecast time the network reads $X_{t-24}, X_{t-12}, X_t$ and predicts maps at $t+12, t+24, t+36, t+48$ hours. Each forecast step adds a learned change to its input map. The next step takes that prediction as input.

## Conventional LSTM

Flatten a map into the 385-element vector $x_k$. At step $k$, concatenate it with the previous 24-element hidden state $h_{k-1}$. One learned affine transformation produces four blocks of gate inputs. Here $\sigma$ is the logistic function, brackets concatenate arrays and $\odot$ multiplies element by element.

$$
(a_i,a_f,a_g,a_o)=W[x_k,h_{k-1}]+b
$$

$$
c_k=\sigma(a_f)\odot c_{k-1}+\sigma(a_i)\odot\tanh(a_g),
\qquad h_k=\sigma(a_o)\odot\tanh(c_k)
$$

$$
\widehat X_{k+1}=X_k+\mathrm{reshape}(Vh_k+d)
$$

Here $X_k$ denotes the map supplied at step $k$. A dense readout predicts a change at every location. The cell and hidden state start at zero.

## Both PredRNN models

The two-layer network keeps the grid intact. Each layer has eight channels, a temporal memory $C$, and a spatial memory $M$ that flows from layer 1 to layer 2 and back to layer 1 at the next step. Write $z$ for the current layer input and $H^-$ for its previous hidden state. Each memory $S$ uses the same update rule with its own learned convolution.

$$
(a_f,a_i,a_g)_C=K_C*[z,H^-]+b_C,\qquad
(a_f,a_i,a_g)_M=K_M*[z,M^-]+b_M
$$

$$
\Delta S=\sigma(a_i)\odot\tanh(a_g),\qquad
S=\sigma(a_f+1)\odot S^-+\Delta S,\qquad S\in\{C,M\}
$$

$$
H=\sigma(K_o*[z,H^-,C,M]+b_o)\odot\tanh(K_r*[C,M]+b_r)
$$

$$
\widehat X_{k+1}=X_k+K_y*H^{(2)}+b_y
$$

The star denotes convolution. The two memory updates use 3 × 3 neighborhoods. The output gate, memory readout and final map readout use 1 × 1 kernels. Layer 2 receives the hidden map from layer 1. All memories start at zero. The constrained model uses these same forecast equations and changes the training objective.

# What the physical constraints measure

Both diagnostics are computed at 500 hPa, after converting the standardized fields back to physical units. The forecast and ERA5 undergo the same calculation at each of the 15 interior cells. This makes the target the balance resolved by ERA5 on this coarse grid, including its nonzero residuals.

## Mass continuity

For eastward wind $u$, northward wind $v$, pressure vertical velocity $\omega=dp/dt$, and latitude $\phi$, the mass-continuity diagnostic is

$$
B_m=D_xu+\frac{D_y(v\cos\phi)}{\cos\phi}+D_p\omega .
$$

The first two terms measure horizontal spreading of air. The last measures how vertical motion varies with pressure. Their sum has units of inverse seconds. For an interior grid cell $(i,j)$, the code uses centered differences with angles in radians and Earth radius $R=6{,}371{,}000$ m.

$$
D_x a=\frac{a_{i,j+1}-a_{i,j-1}}{2R\cos\phi_i\,\Delta\lambda},
\qquad D_y a=\frac{a_{i+1,j}-a_{i-1,j}}{2R\,\Delta\phi}
$$

$$
D_p a=\frac{a_{600}-a_{400}}{20{,}000\ {\mathrm{Pa}}}
$$

Here $\lambda$ and $\phi$ are longitude and latitude. The 400 and 600 subscripts denote pressure levels in hPa. One grid cell around the scoring area supplies the neighboring values for the spatial differences.

## Dry static energy

Dry static energy per unit mass is $s=c_p T+\Phi$, where $T$ is temperature in kelvins, $\Phi$ is geopotential in J/kg and $c_p=1{,}004$ J/(kg K). Its diagnostic combines the local 12-hour change with horizontal and vertical transport.

$$
B_e=\frac{s_k-s_{k-1}}{43{,}200\ {\mathrm{s}}}
    +u_kD_xs_k+v_kD_ys_k+\omega_kD_p s_k .
$$

The resulting tendency has units of W/kg and can reflect heating and unresolved processes. At the first forecast lead, $s_{k-1}$ comes from the last observed input. At subsequent leads the forecast diagnostic uses the preceding forecast, and the ERA5 diagnostic uses the preceding ERA5 map.

# What each model minimizes

The common forecast loss $L_f$ measures error in all 11 standardized fields over the full 5 × 7 map. Angle brackets denote an arithmetic mean over the indicated dimensions. The small constant makes the square root well behaved near zero.

$$
L_f=\sqrt{\left\langle(\widehat X-X)^2\right\rangle_{\mathrm{batch,lead,field,grid}}+10^{-12}}
$$

Both PredRNN models also penalize alignment between the two memory writes. A learned, shared 1 × 1 projection $A$ maps each write into channels. The cosine compares the flattened spatial maps within each channel. Its absolute value is averaged over samples, channels, layers and recurrent steps.

$$
L_d=\left\langle\left|\cos(A\Delta C,A\Delta M)\right|\right\rangle
$$

For diagnostic $q$ (mass or energy), $\tau_q$ is the training standard deviation of the ERA5 diagnostic, floored at $10^{-12}$. The area weight $w_i=\cos\phi_i/\mathrm{mean}_{\mathrm{interior}}(\cos\phi)$ has mean one. The penalty measures the squared, scaled discrepancy on the interior.

$$
P_q=\left\langle w_i\left(\frac{B_q(\widehat X)-B_q(X)}{\tau_q}\right)^2
       \right\rangle_{\mathrm{batch,lead,interior}},\qquad q\in\{m,e\}
$$

These terms give the three training objectives.

$$
L_{\mathrm{LSTM}} = L_f
$$

$$
L_{\mathrm{PredRNN}} = L_f+0.1L_d
$$

$$
L_{\mathrm{constrained}} = L_f+0.1L_d+\lambda_m P_m+\lambda_e P_e
$$

The selected ten-year weights were $\lambda_m=0.030$ and $\lambda_e=0.003$. The penalties act during training. At forecast time the constrained network uses only its inputs and learned weights. During training both PredRNNs mix observed and generated frames by the same schedule. Across epochs the observed-frame probability rises from 0.5 to 1 for context frames after the first, and falls from 0.5 to 0 for decoder inputs.

## How the paper scores wind forecasts

At lead $h$, let $\delta u$ and $\delta v$ be the forecast errors in m/s. For each seed, the score averages across issue dates and interior grid cells. The tables then average the three seed-specific scores. A positive physics gain means that adding both penalties reduced wind error.

$$
R_h=\sqrt{\left\langle w_i[(\delta u_h)^2+(\delta v_h)^2]\right\rangle_{\mathrm{date,interior}}},
\qquad G_h=\mathrm{mean}_{\mathrm{seeds}}(R_{h,\mathrm{plain}}-R_{h,\mathrm{constrained}})
$$
