using Plots
using LaTeXStrings
using LinearAlgebra
using Turing
using ReverseDiff
using BenchmarkTools
using Turing
using PreallocationTools: DiffCache, get_tmp
using AdvancedHMC


# using FFTW

include("fft.jl")

function λ(E::T)::T where T <: Real
    hc::T = 12.398 #keV*Angstrom
    E₀::T = 511.0 #keV
    return hc/sqrt(E*(2*E₀ + E))
end


function Δf(E::T, Cs::T, nᵢ=1)::T where T <: Real
    return sqrt((2*nᵢ-0.5)*Cs*λ(E))
end
kmax(E, Cs) = (6/((Cs*λ(E)^3)))^(1/4)
σ(E::Real) = 2π/(λ(E)*E) * (511 + E)/(2*511 + E) # per Angstroms

Diff_Image(image, c=0.1) = @. log1p(c*abs(image))

E = 300.0 #keV
Cs = 1 #mm
Cs *= 1e7 #convert to Angstroms

N = 2^5

upper = (N-1)/2 + 0.1
lower = (-N+1)/2
kx = collect(lower:1:upper)
ky = collect(lower:1:upper)

k = @. sqrt(kx'^2 + ky^2)
disk = k.<=N/2

kx = kx./maximum(kx).*kmax(E, Cs)
ky = ky./maximum(ky).*kmax(E, Cs)
k = @. sqrt(kx'^2 + ky^2)



Δk = kx[2] - kx[1]
L = 1/Δk
ΔL = L/N

x = collect(range(-L/2, L/2, N))
y = collect(range(-L/2, L/2, N))


χ(k) =@. π*λ(E)*k^2*(0.5*Cs*λ(E)^2*k^2 - Δf(E, Cs))
H = @. exp(-1im*χ(k)*disk)*disk



function Si3N4_Grating_Parameters(E::T) where T <: Real
    #given from CJW thesis in units of per nm
    σU = 0.15 
    α = 0.008
    
    

    λ_nm = λ(E)/10 # convert to nm from Angstroms
    κ = α * (λ_nm/(2*π)) # this is the attenuation coefficient 

    # I assume that CJW uses 300 KeV for his values
    σ_cjw = σ(E) * 10 # per nm 
    U_mip = σU/σ_cjw 
    return (U_mip, κ)
end

function Blazed_Grating(p; N=N, mill_depth=208, thickness=500, E::Real=E)
    """
    Create a blazed grating transmission function for a given energy E.
    """
    
    U_mip, κ = Si3N4_Grating_Parameters(E)

    """
    creatings the 1D grating function (depth(x)), slopes upwards to the right
    """
    xs = collect(range(0, 1, length=N))
    d0 = thickness - mill_depth
    g_1d = @. mod(p*mill_depth*xs, mill_depth+1) + d0



    t_1d = @. cis((σ(E)*U_mip+1im*κ*2π/λ(E))*g_1d);
    t_2d = repeat(t_1d', N, 1)
    return t_2d
end
t1 = Blazed_Grating(5)
data = abs2.(fftnshift(t1.*H))
using StaticArrays
@model function single_propagation(y::AbstractMatrix{<:AbstractFloat}, 
                                    params::Vector{<:AbstractFloat},
                                    arrs::Vector{<:Matrix{<:AbstractFloat}},
                                    disk::BitMatrix,
                                    caches::Vector{<:DiffCache};
                                    dist::Distribution)
    Cs_ctr, E_ctr, Δf_ctr, Cs_σ, E_σ, Δf_σ = params
    t1r, t1i, k = arrs

    Cs ~ dist
    E ~ dist
    Δf ~ dist
    # Cs ~ Normal(0,1)
    # E ~ Normal(0,1)
    # Δf ~ Normal(0,1)
    # MvNormal(zeros(3), I)
    Cs = Cs_ctr + Cs_σ * Cs
    E = E_ctr + E_σ * E
    Δf = Δf_ctr + Δf_σ * Δf

    # Hr, Hi = get_tmp(caches[1],E), get_tmp(caches[2],E)
    # abberations = get_tmp(caches[3],E)
    
    prodr = get_tmp(caches[1],E)
    prodi = get_tmp(caches[2],E)
    buff = get_tmp(caches[3],E)


    # pieλ = π*λ(E)
    # Csλ2 = 0.5*Cs*λ(E)^2
    # @inbounds @simd for i in eachindex(k)
    #     # abberations[i] = π*λ(E)*k[i]^2*(0.5*Cs*λ(E)^2*k[i]^2 - Δf) * disk[i]
    #     # Hi[i], Hr[i] = sincos(-abber)
    #     # Hi[i], Hr[i] = Hi[i]*disk[i], Hr[i]*disk[i]
    #     k2 = k[i]^2
    #     abber = pieλ*k2*(Csλ2*k2 - Δf) * disk[i]
    #     Hi, Hr = sincos(-abber)
    #     # Hi, Hr = Hi*disk[i], Hr*disk[i]

    #     prodr[i] = (t1r[i] * Hr - t1i[i] * Hi) * disk[i]
    #     prodi[i] = (t1r[i] * Hi + t1i[i] * Hr) * disk[i]
    # end
    # @inbounds @simd for i in eachindex(k)
    #     # abberations[i] = π*λ(E)*k[i]^2*(0.5*Cs*λ(E)^2*k[i]^2 - Δf) * disk[i]
    #     # Hi[i], Hr[i] = sincos(-abber)
    #     # Hi[i], Hr[i] = Hi[i]*disk[i], Hr[i]*disk[i]
        
    #     abber = π*λ(E)*k[i]^2*(0.5*Cs*λ(E)^2*k[i]^2 - Δf) * disk[i]
    #     Hi, Hr = sincos(-abber)
    #     # Hi, Hr = Hi*disk[i], Hr*disk[i]

    #     prodr[i] = (t1r[i] * Hr - t1i[i] * Hi) * disk[i]
    #     prodi[i] = (t1r[i] * Hi + t1i[i] * Hr) * disk[i]
    # end
    # disk_indices = findall(disk)
    pieλ = π*λ(E)
    Csλ2 = Cs/2*λ(E)^2
    @inbounds @simd for i in eachindex(k)
        if disk[i] 
            k2 = k[i]^2
            abber = pieλ*k2*(Csλ2*k2 - Δf)
            Hi, Hr = sincos(-abber)

            prodr[i] = t1r[i] * Hr - t1i[i] * Hi
            prodi[i] = t1r[i] * Hi + t1i[i] * Hr
        end
    end
    # @inbounds @simd for i in findall(disk)
        
    #     k2 = k[i]^2
    #     abber = pieλ*k2*(Csλ2*k2 - Δf)
    #     Hi, Hr = sincos(-abber)

    #     prodr[i] = t1r[i] * Hr - t1i[i] * Hi
    #     prodi[i] = t1r[i] * Hi + t1i[i] * Hr
    # end
    
    fftshift!(prodr, buff)
    fftshift!(prodi, buff)
    fft!(prodr, prodi)
    fftshift!(prodr, buff)
    fftshift!(prodi, buff)
    
    # @. H = abs2($fftnshift(abberations))


    # χ(k) =@. π*λ(E)*k^2*(0.5*Cs*λ(E)^2*k^2 - Δf(E, Cs))
    # H = @. phonycomplex(cos(χ(k)*disk)*disk, sin(-χ(k)*disk)*disk);
    # H = @. exp(-1im*χ(k)*disk)*disk
    @inbounds @simd for i in eachindex(k)
        # buff[i] = prodr[i]^2 + prodi[i]^2
        intermediate = prodr[i] * prodr[i] + prodi[i] * prodi[i]
        Turing.@addlogprob! -(y[i] - intermediate)^2 
        
        # Turing.@addlogprob! -(y[i] - buff[i])^2 
        # Turing.@addlogprob! data[i]log(model[i]) - model[i] #poissonian
    end
end

stds = [5*1e6, 3, 1e2]
params0 = [Cs+1e6, E+2.0, Δf(E, Cs)-2*1e1]
params = [params0..., stds...];

arrs = [real.(t1), imag.(t1), k]
caches = [DiffCache(zeros(size(k))) for _ in 1:3]
model = single_propagation(data, params, arrs, disk, caches; dist=Turing.Normal(0, 1));


δ_target = 0.65
ϵ0 = 0.1
D = 3
metric = DenseEuclideanMetric(D)
kernel = HMCKernel(Trajectory{MultinomialTS}(Leapfrog(ϵ0), GeneralisedNoUTurn(max_depth=9)))
mma = MassMatrixAdaptor(metric)
ssa = StepSizeAdaptor(δ_target, Leapfrog(ϵ0))
adaptor = StanHMCAdaptor(mma, ssa)
hmcsampler = AdvancedHMC.HMCSampler(kernel, metric, adaptor)  
nuts = externalsampler(hmcsampler)

n_adapts = 1000
n_samples = n_adapts + 0
n_chains = 2
initial_params = [clamp.(randn(3),-1.,1.) for _ in 1:n_chains]

best_lp = -Inf
best_res = []
best_params = []
for init_params in initial_params
    res = Turing.Optimisation.maximum_a_posteriori(model; initial_params=init_params)
    println("MAP estimate: ", res, " with logprob: ", res.lp)
    if res.lp > best_lp
        global best_lp = res.lp
        global best_params = res.values.array
        global best_res = res
    end
end

println("Best MAP estimate: ", best_params, " with logprob: ", best_lp)
initial_params = [best_params for _ in 1:n_chains];
using Profile
using PProf
@benchmark sample($model,$nuts,
                    $MCMCThreads(), 
                    $n_samples, $n_chains; 
                    progress=false, 
                    $n_adapts, 
                    $initial_params,
                    save_state=true, 
                    adtype = AutoReverseDiff(true))
# @profview_allocs sample(model,nuts,
#                     MCMCThreads(), 
#                     n_samples, n_chains; 
#                     progress=false, 
#                     n_adapts, 
#                     initial_params,
#                     save_state=true, 
#                     adtype = AutoReverseDiff(true))

# Profile.Allocs.clear()
# Profile.Allocs.@profile sample(model,nuts,
#                     MCMCThreads(), 
#                     n_samples, n_chains; 
#                     progress=false, 
#                     n_adapts, 
#                     initial_params,
#                     save_state=true, 
#                     adtype = AutoReverseDiff(true))
# PProf.Allocs.pprof(from_c=false)