function transmission(g::AbstractMatrix, aperature::Union{AbstractMatrix, Bool}=true; 
            σU::Real=π/21, α::Real=0.008)
    # mask selection: matrix aperture, disk when true, or no mask when false
    mask = aperature isa AbstractMatrix ? aperature :
           aperature ? centered_disks(size(g,1)) :
           ones(eltype(g), size(g))

    # apply transmission (preserves previous behavior; pass σU=0.15 if you need that legacy value)
    t = @. cis((σU + 1im * α) * g / 10 * mask) * mask
    return t
end

function transmission(g::AbstractVector; 
            σU::Real=π/21, α::Real=0.008)

    # apply transmission (preserves previous behavior; pass σU=0.15 if you need that legacy value)
    t = @. cis((σU + 1im * α) * g / 10 )
    return t
end

function blazed_grating(N, p; thickness=500, mill_depth=208)
    """
        thickness = 500 #Angstroms
        mill_depth = 208 #Angstroms
    """
    @assert N % p == 0 "p must divide N"

    Ny, Nx = N, N

    xs = (0:Nx-1) ./ Nx                # N points, endpoint 1 is excluded
    d0 = thickness - mill_depth
    g1 = @. mill_depth*mod(p*xs, 1) + d0
    
    g2 = repeat(g1', N, 1)
    return g2
end

function blazed_grating(N, p, x_apex; thickness=500, mill_depth=208)
    """
        thickness = 500 #Angstroms
        mill_depth = 208 #Angstroms
    """
    @assert N % p == 0 "p must divide N"
    Ny, Nx = N, N


    # xs = (0:(Nx÷p)-1) ./ (Nx÷p)                # N/p points, endpoint 1 is excluded
    # d0 = thickness - mill_depth
    # g1 = zeros(Float64, Nx÷p)
    # for (i,x) in enumerate(xs)
    #     if x <= x_apex
    #         g1[i] = mill_depth/x_apex * x  + d0
    #     else
    #         g1[i] = -mill_depth/(1-x_apex) * (x-x_apex) + thickness 
    #     end
    # end


    xs = collect(Int, 0:1:(Nx÷p)-1)                # N/p points, endpoint 1 is excluded
    x_apex = floor(Int, x_apex * xs[end])

    d0 = thickness - mill_depth
    g1 = zeros(Float64, Nx÷p)
    for (i,x) in enumerate(xs)
        if x <= x_apex
            g1[i] = x/(x_apex+1)
        else
            x_shifted = x - (x_apex + 1)
            g1[i] = -x_shifted/(xs[end] - x_apex) + oneunit(Int)
        end
    end
    g1 = @. g1 * mill_depth + d0
    g2 = repeat(g1', N, p)
    return g2
end

function blazed_grating_1d(N, p, x_apex; thickness=500, mill_depth=208)
    """
        thickness = 500 #Angstroms
        mill_depth = 208 #Angstroms
    """
    @assert N % p == 0 "p must divide N"
    
    xs = collect(Int, 0:1:(N÷p)-1)                # N/p points, endpoint 1 is excluded
    x_apex = floor(Int, x_apex * xs[end])
    d0 = thickness - mill_depth
    g1 = zeros(Float64, N÷p)
    for (i,x) in enumerate(xs)
        if x <= x_apex
            g1[i] = x/(x_apex+1)
        else
            x_shifted = x - (x_apex + 1)
            g1[i] = -x_shifted/(xs[end] - x_apex) + oneunit(Int)
        end
    end
    g1 .= g1 .* mill_depth .+ d0
    g1 = repeat(g1, p)
    return g1
end


function tanh_values(N, tanh_slope_cutoff, k)
    x_cutoff_half_width = asech(sqrt(2*tanh_slope_cutoff/k))/k
    x_shift = 1 - x_cutoff_half_width
    xs = (0:N-1) ./ N
    tanh_vals = @. (1/2) * (tanh(-k * (xs - x_shift )) + 1) #smooth flipped heaviside function
    return tanh_vals, 2*x_cutoff_half_width
end

function tanh_blazed_grating(N, p, x_apex, k; thickness=500, mill_depth=208)
    """
        thickness = 500 #Angstroms
        mill_depth = 208 #Angstroms
    """
    @assert N % p == 0 "p must divide N"
    Ny, Nx = N, N

    tanh_slope_cutoff = 0.1
    tanh_vals, x_cutoff_width = tanh_values(Nx÷p, tanh_slope_cutoff, k)
    
    @assert x_cutoff_width < 1 - x_apex "k is too small or change cutoff value"

    

    xs = collect(Int, 0:1:(Nx÷p)-1)                # N/p points, endpoint 1 is excluded
    x_apex = floor(Int, x_apex * xs[end])
    d0 = thickness - mill_depth
    g1 = zeros(Float64, Nx÷p)
    for (i,x) in enumerate(xs)
        if x <= x_apex
            g1[i] = x/(x_apex+1)
        else
            g1[i] = tanh_vals[i]
        end
    end
    g1 = @. g1 * mill_depth + d0

    g2 = repeat(g1', N, p)
    return g2
end

function sinusoidal_grating(N, p; thickness=500, mill_depth=208)
    """
        thickness = 500 #Angstroms
        mill_depth = 208 #Angstroms
    """
    @assert N % p == 0 "p must divide N"
    Ny, Nx = N, N

    xs = (0:Nx-1) ./ Nx                # N points, endpoint 1 is excluded
    d0 = thickness - mill_depth
    g1 = @. (mill_depth/2)*(1 .+ sin.(2π*p*xs .- π/2)) + d0
    
    g2 = repeat(g1', N, 1)
    return g2
end