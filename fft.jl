function fft!(x_re::AbstractVector{T}, x_im::AbstractVector{T}) where {T<:Any}
    N = length(x_re)
    @assert N == length(x_im) "re/im must match length"
    @assert ispow2(N) "Length must be a power of two"

    # Bit‑reversal on both real & imag
    bits = Int(floor(log2(N)))
    @inbounds for i in 0:N-1
        j = bitreverse(i) >>> (sizeof(Int)*8 - bits)
        if j > i
            x_re[i+1], x_re[j+1] = x_re[j+1], x_re[i+1]
            x_im[i+1], x_im[j+1] = x_im[j+1], x_im[i+1]
        end
    end

    # Cooley‑Tuk stages
    m = 1
    while m < N
        half = m
        m *= 2
        θ = -2π / m
        @inbounds for j in 0:half-1
            c = cos(θ*j)     # real constant :contentReference[oaicite:8]{index=8}
            s = sin(θ*j)
            for k in j:m:N-1
                u = k + 1
                v = k + half + 1
                u_re, u_im = x_re[u], x_im[u]
                v_re, v_im = x_re[v], x_im[v]

                # complex multiply (v * cis(θ*j)) in real form
                t_re =  c*v_re - s*v_im
                t_im =  s*v_re + c*v_im

                # butterfly
                x_re[u] = u_re + t_re
                x_im[u] = u_im + t_im
                x_re[v] = u_re - t_re
                x_im[v] = u_im - t_im
            end
        end
    end

    return nothing
end
function fft!(A_re::AbstractMatrix{T}, A_im::AbstractMatrix{T}) where {T<:Any}
    Ny, Nx = size(A_re)
    @assert size(A_im) == (Ny, Nx) "Real and imaginary parts must have same dimensions"
    @assert ispow2(Ny) && ispow2(Nx) "Both dimensions must be powers of 2"

    # 1) FFT down each column
    @inbounds for x in 1:Nx
        # Get column views
        col_re = @view A_re[:, x]
        col_im = @view A_im[:, x]
        # In-place FFT on column
        fft!(col_re, col_im)
    end

    # 2) FFT across each row
    @inbounds for y in 1:Ny
        # Get row views
        row_re = @view A_re[y, :]
        row_im = @view A_im[y, :]
        # In-place FFT on row
        fft!(row_re, row_im)
    end
    
    return nothing
end
function fftshift!(x::AbstractArray, tmp::AbstractArray)
    # Calculate the shift amount for each dimension
    shifts = map(s -> div(s, 2), size(x))
    circshift!(tmp,x, shifts)
    # x = tmp    
    copyto!(x, tmp)
    return nothing
end

# Inverse operation
function ifftshift!(x::AbstractArray,tmp::AbstractArray)
    shifts = map(s -> -div(s, 2), size(x))
    circshift!(tmp, x, shifts)
    copyto!(x, tmp)
    # x = tmp
    return nothing
end






function fft(x::AbstractVector)
    N = length(x)
    @assert ispow2(N) "Length must be a power of 2"
    # Make a working copy
    y = complex.(copy(x))

    # 1) Bit‑reversal permutation
    bits = Int(floor(log2(N)))
    for i in 0:(N-1)
        j = bitreverse(i) >>> (sizeof(Int)*8 - bits)
        if j > i
            y[i+1], y[j+1] = y[j+1], y[i+1]
        end
    end

    # 2) Cooley‑Tuk decimation‑in‑place
    m = 1
    while m < N
        half = m
        m *= 2
        # principal m‑th root of unity: exp(-2πi/m)
        wm = cis(-2π/m)
        for k in 1:m:N
            w = one(wm)
            for j in 0:(half-1)
                u = y[k + j]
                t = w * y[k + j + half]
                y[k + j]          = u + t
                y[k + j + half]   = u - t
                w *= wm
            end
        end
    end

    return y
end

function ifft(x::AbstractVector)
    N = length(x)
    @assert ispow2(N) "Length must be a power of 2"
    # Make a working copy
    y = complex.(copy(x))

    # 1) Bit‑reversal permutation
    bits = Int(floor(log2(N)))
    for i in 0:(N-1)
        j = bitreverse(i) >>> (sizeof(Int)*8 - bits)
        if j > i
            y[i+1], y[j+1] = y[j+1], y[i+1]
        end
    end

    # 2) Cooley‑Tuk decimation‑in‑place
    m = 1
    while m < N
        half = m
        m *= 2
        # principal m‑th root of unity: exp(-2πi/m)
        wm = cis(2π/m)
        for k in 1:m:N
            w = one(wm)
            for j in 0:(half-1)
                u = y[k + j]
                t = w * y[k + j + half]
                y[k + j]          = u + t
                y[k + j + half]   = u - t
                w *= wm
            end
        end
    end

    return y
end

function fft(A::AbstractMatrix)
    Ny, Nx = size(A)
    @assert ispow2(Ny) && ispow2(Nx) "Both dimensions must be powers of 2"
    B = copy(A)

    # 1) FFT down each column
    @inbounds @simd for x in 1:Nx
        B[:, x] = fft(view(B, :, x))
    end

    # 2) FFT across each row of the intermediate result
    @inbounds @simd for y in 1:Ny
        B[y, :] = fft(view(B, y, :))
    end
    return B
end

function ifft(A::AbstractMatrix)
    Ny, Nx = size(A)
    @assert ispow2(Ny) && ispow2(Nx) "Both dimensions must be powers of 2"
    B = copy(A)

    # 1) FFT down each column
    @inbounds @simd for x in 1:Nx
        B[:, x] = ifft(view(B, :, x))
    end

    # 2) FFT across each row of the intermediate result
    @inbounds @simd for y in 1:Ny
        B[y, :] = ifft(view(B, y, :))
    end
    return B
end

function fftshift(x::AbstractArray)
    # Calculate the shift amount for each dimension
    shifts = map(s -> div(s, 2), size(x))    
    return circshift(x, shifts)
end
function ifftshift!(x::AbstractArray)
    shifts = map(s -> -div(s, 2), size(x))
    return circshift(x, shifts)
end


function fftnshift(A::AbstractArray)
    B = copy(A)
    return fftshift(fft(fftshift(B)))
end
function ifftnshift(A::AbstractArray)
    B = copy(A)
    return ifftshift(ifft(ifftshift(B)))./(reduce(*,size(B)))
end