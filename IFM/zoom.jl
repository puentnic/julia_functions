function zoom_heatmap(x::AbstractVector, y::AbstractVector, image::AbstractMatrix, width=50; kwargs...)
    N = size(image, 1)
    center = div(N, 2) + 1
    half_width = div(width, 2)
    
    x_zoomed = x[center-half_width:center+half_width]
    y_zoomed = y[center-half_width:center+half_width]
    image_zoomed = image[center-half_width:center+half_width, center-half_width:center+half_width]
    
    return heatmap(x_zoomed, 
                    y_zoomed, 
                    image_zoomed; 
                    xlims=(x_zoomed[1], x_zoomed[end]), 
                    ylims=(y_zoomed[1],y_zoomed[end]), 
                    kwargs...
                    )
end
function zoom_heatmap(image::AbstractMatrix, width=50; kwargs...)
    Ny, Nx = size(image)
    ctrx::Int32 = div(Nx, 2) + 1
    ctry::Int32 = div(Ny, 2) + 1
    half_width::Int32 = div(width, 2)


    x = collect(1:Nx)
    y = collect(1:Ny)
    x_zoomed = x[ctrx-half_width:ctrx+half_width]
    y_zoomed = y[ctry-half_width:ctry+half_width]
    image_zoomed = image[ctry-half_width:ctry+half_width, ctrx-half_width:ctrx+half_width]
    
    return heatmap(x_zoomed, 
                    y_zoomed, 
                    image_zoomed; 
                    xlims=(x_zoomed[1], x_zoomed[end]), 
                    ylims=(y_zoomed[1],y_zoomed[end]), 
                    kwargs...
                    )
end
function zoom_plot(x::AbstractVector, y::AbstractVector, width = 20; kwargs...)
    @assert length(x) == length(y) "x and y must have the same length"
    N = length(y)
    ctr = N÷2 + 1
    half_width = width ÷ 2
    x_zoomed = x[ctr-half_width:ctr+half_width]
    y_zoomed = y[ctr-half_width:ctr+half_width]

    return plot(x_zoomed, y_zoomed; 
            xlims=(x_zoomed[1], x_zoomed[end]), 
            ylims=(y_zoomed[1], y_zoomed[end]), 
            kwargs...
            )
end
function zoom_plot(y::AbstractVector, width = 20; kwargs...)
    N = length(y)
    ctr = N÷2 + 1
    half_width = width ÷ 2
    x = collect(1:N)
    x_zoomed = x[ctr-half_width:ctr+half_width]
    y_zoomed = y[ctr-half_width:ctr+half_width]
    return plot(x_zoomed, y_zoomed; 
            xlims=(x_zoomed[1], x_zoomed[end]), 
            ylims=(y_zoomed[1], y_zoomed[end]),
            kwargs...
            )
end