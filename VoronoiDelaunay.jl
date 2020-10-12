module VoronoiDelaunay

export
DelaunayTessellation, DelaunayTessellation2D, sizehint!, isexternal,
min_coord, max_coord, locate, movea, moveb, movec,
delaunayedges, voronoiedges, voronoiedgeswithoutgenerators, voronoicells,
iterate, findindex, push!,
Point, Point2D, AbstractPoint2D, getx, gety, geta, getb, getc,
getgena, getgenb, getgen, getplotxy, getplotcolors

using GeometricalPredicates
import GeometricalPredicates: geta, getb, getc

import Base: push!, iterate, copy, sizehint!
import Colors: RGB, RGBA
using Random: shuffle!

const scale_down=0.9
const width = (GeometricalPredicates.max_coord-GeometricalPredicates.min_coord-2*eps(Float64))*scale_down
const min_coord = 1.5-width/2
const max_coord = 1.5+width/2
const min_coord1 = GeometricalPredicates.min_coord+eps(Float64)
const max_coord1 = GeometricalPredicates.max_coord-eps(Float64)

mutable struct DelaunayTriangle{T<:AbstractPoint2D} <: AbstractNegativelyOrientedTriangle
    _a::T; _b::T; _c::T
    _bx::Float64; _by::Float64
    _cx::Float64; _cy::Float64
    _px::Float64; _py::Float64;_pr2::Float64
    _neighbour_a::Int64
    _neighbour_b::Int64
    _neighbour_c::Int64

    function DelaunayTriangle{T}(pa::T, pb::T, pc::T,
                                na::Int64, nb::Int64, nc::Int64) where T
        t = new(pa, pb, pc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, na, nb, nc)
        clean!(t)
        t
    end

    # this constructor makes copies
    function DelaunayTriangle{T}(pa::T, pb::T, pc::T,
                                bx::Float64, by::Float64,
                                cx::Float64, cy::Float64,
                                px::Float64, py::Float64,
                                pr2::Float64,
                                na::Int64, nb::Int64, nc::Int64) where T
        new(pa, pb, pc, bx, by, cx, cy, px, py, pr2, na, nb, nc)
    end
end
function DelaunayTriangle(pa::T, pb::T, pc::T,
                        na::Int64, nb::Int64, nc::Int64) where T<:AbstractPoint2D
    DelaunayTriangle{T}(pa, pb, pc, na, nb, nc)
end
function DelaunayTriangle(pa::T, pb::T, pc::T,
                          bx::Float64, by::Float64,
                          cx::Float64, cy::Float64,
                          px::Float64, py::Float64,
                          pr2::Float64,
                          na::Int64, nb::Int64, nc::Int64) where T<:AbstractPoint2D
    DelaunayTriangle{T}(pa, pb, pc, bx, by, cx, cy, px, py, pr2, na, nb, nc)
end

function copy(t::DelaunayTriangle{T}) where T<:AbstractPoint2D
    DelaunayTriangle(
                     t._a, t._b, t._c,
                     t._bx, t._by,
                     t._cx, t._cy,
                     t._px, t._py,
                     t._pr2,
                     t._neighbour_a, t._neighbour_b, t._neighbour_c
                     )
end

function isexternal(t::DelaunayTriangle{T}) where T<:AbstractPoint2D
    getx(geta(t)) < min_coord1 || getx(geta(t)) > max_coord1 ||
    getx(getb(t)) < min_coord1 || getx(getb(t)) > max_coord1 ||
    getx(getc(t)) < min_coord1 || getx(getc(t)) > max_coord1
end

mutable struct DelaunayTessellation2D{T<:AbstractPoint2D}
    _trigs::Vector{DelaunayTriangle{T}}
    _last_trig_index::Int64
    _edges_to_check::Vector{Int64}
    _total_points_added::Int64
    _IO::String

    function DelaunayTessellation2D{T}(n::Int64 = 100) where T
        a = T(GeometricalPredicates.min_coord, GeometricalPredicates.min_coord)
        b = T(GeometricalPredicates.min_coord, GeometricalPredicates.max_coord)
        c = T(GeometricalPredicates.max_coord, GeometricalPredicates.min_coord)
        d = T(GeometricalPredicates.max_coord, GeometricalPredicates.max_coord)
        t1 = DelaunayTriangle{T}(d,c,b, 2,1,1)
        t2 = DelaunayTriangle{T}(a,b,c, 3,1,1)
        t3 = DelaunayTriangle{T}(d,c,b, 2,1,1)
        _trigs = DelaunayTriangle{T}[t1, t2, t3]
        t = new(_trigs, 3, Int64[], 0, "Inside")
        sizehint!(t._edges_to_check, 1000)
        sizehint!(t, n)
    end
end
DelaunayTessellation2D(n::Int64) = DelaunayTessellation2D{Point2D}(n)
DelaunayTessellation2D(n::Int64, ::T) where {T<:AbstractPoint2D} = DelaunayTessellation2D{T}(n)
DelaunayTessellation(n::Int64=100) = DelaunayTessellation2D(n)

function sizehint!(t::DelaunayTessellation2D{T}, n::Int64) where T<:AbstractPoint2D
    required_total_size = 2n + 10
    required_total_size <= length(t._trigs) && return
    sizehint!(t._trigs, required_total_size)
    while length(t._trigs) < required_total_size
        push!(t._trigs, copy(t._trigs[end]))
    end
    t
end

# growing strategy
function sizefit_at_least(t::DelaunayTessellation2D{T}, n::Int64) where T<:AbstractPoint2D
    minimal_acceptable_actual_size = 2*n+10
    minimal_acceptable_actual_size <= length(t._trigs) && return
    required_total_size = length(t._trigs)
    while required_total_size < minimal_acceptable_actual_size
        required_total_size += required_total_size >>> 1
    end
    sizehint!(t._trigs, required_total_size)
    while length(t._trigs) < required_total_size
        push!(t._trigs, copy(t._trigs[end]))
    end
    t
end

struct DelaunayEdge{T<:AbstractPoint2D}
    _a::T
    _b::T
end
geta(e::DelaunayEdge{T}) where {T<:AbstractPoint2D} = e._a
getb(e::DelaunayEdge{T}) where {T<:AbstractPoint2D} = e._b

struct VoronoiEdge{T<:AbstractPoint2D}
    _a::Point2D
    _b::Point2D
    _generator_a::T
    _generator_b::T
end
geta(e::VoronoiEdge{T}) where {T<:AbstractPoint2D} = e._a
getb(e::VoronoiEdge{T}) where {T<:AbstractPoint2D} = e._b
getgena(e::VoronoiEdge{T}) where {T<:AbstractPoint2D} = e._generator_a
getgenb(e::VoronoiEdge{T}) where {T<:AbstractPoint2D} = e._generator_b


struct VoronoiEdgeWithoutGenerators
    _a::Point2D
    _b::Point2D
end
geta(e::VoronoiEdgeWithoutGenerators) = e._a
getb(e::VoronoiEdgeWithoutGenerators) = e._b


#Needed for plotting
struct VoronoiCell{T<:AbstractPoint2D}
    _summits::Vector{Point2D}
    _generator::T
end
getsummits(e::VoronoiCell{T}) where {T<:AbstractPoint2D} = e._summits
getgen(e::VoronoiCell{T}) where {T<:AbstractPoint2D} = e._generator


function delaunayedges(t::DelaunayTessellation2D)
    visited = zeros(Bool, t._last_trig_index)
    function delaunayiterator(c::Channel)
        @inbounds for ix in 2:t._last_trig_index
            tr = t._trigs[ix]
            isexternal(tr) && continue
            visited[ix] && continue
            visited[ix] = true
            ix_na = tr._neighbour_a
            if !visited[ix_na]
                put!(c, DelaunayEdge(getb(tr), getc(tr)))
            end
            ix_nb = tr._neighbour_b
            if !visited[ix_nb]
                put!(c, DelaunayEdge(geta(tr), getc(tr)))
            end
            ix_nc = tr._neighbour_c
            if !visited[ix_nc]
                put!(c, DelaunayEdge(geta(tr), getb(tr)))
            end
        end
    end
    Channel(delaunayiterator)
end

function isextreme(p::T) where T<:AbstractPoint2D
    if p==Point2D(GeometricalPredicates.min_coord, GeometricalPredicates.min_coord) || p == Point2D(GeometricalPredicates.max_coord, GeometricalPredicates.min_coord) || p == Point2D(GeometricalPredicates.min_coord, GeometricalPredicates.max_coord) || p == Point2D(GeometricalPredicates.max_coord, GeometricalPredicates.max_coord)
        return true
    else
        return false
    end
end
        
        
#the voronoi edges are the segments between two circumcentres, reshaped so it fits inside the square
function voronoiedges(t::DelaunayTessellation2D)
    visited = zeros(Bool, t._last_trig_index)
    visited[1] = true
    function voronoiiterator(c::Channel)
        if t._IO=="Inside"
            for ix in 2:t._last_trig_index
                visited[ix] && continue
                tr = t._trigs[ix]
                visited[ix] = true
                #isexternal(tr) && continue
                cc = circumcenter(tr)

                ix_na = tr._neighbour_a
                if !visited[ix_na] #&& !isexternal(t._trigs[ix_na])
                    cc2=cc
                    nb = t._trigs[ix_na]
                    cc1=circumcenter(nb)
                    x,y,x1,y1=getx(cc2),gety(cc2),getx(cc1),gety(cc1)
                    if x<min_coord
                        s=(min_coord-x)/(x1-x)
                        cc2=Point2D(min_coord,y+s*(y1-y))
                    elseif x>max_coord
                        s=(max_coord-x)/(x1-x)
                        cc2=Point2D(max_coord,y+s*(y1-y))
                    end
                    x,y=getx(cc2),gety(cc2)
                    if y<min_coord
                        s=(min_coord-y)/(y1-y)
                        cc2=Point2D(x+s*(x1-x),min_coord)
                    elseif y>max_coord
                        s=(max_coord-y)/(y1-y)
                        cc2=Point2D(x+s*(x1-x),max_coord)
                    end
                    x,y=getx(cc2),gety(cc2)
                    if x1<min_coord
                        s=(min_coord-x1)/(x-x1)
                        cc1=Point2D(min_coord,y1+s*(y-y1))
                    elseif x1>max_coord
                        s=(max_coord-x1)/(x-x1)
                        cc1=Point2D(max_coord,y1+s*(y-y1))
                    end
                    x1,y1=getx(cc1),gety(cc1)
                    if y1<min_coord
                        s=(min_coord-y1)/(y-y1)
                        cc1=Point2D(x1+s*(x-x1),min_coord)
                    elseif y1>max_coord
                        s=(max_coord-y1)/(y-y1)
                        cc1=Point2D(x1+s*(x-x1),max_coord)
                    end
                    x,y,x1,y1=getx(cc2),gety(cc2),getx(cc1),gety(cc1)
                    if x != x1 || y != y1 || !isextreme(getb(tr)) || !isextreme(getc(tr))
                        put!(c, VoronoiEdge(cc2, cc1, getb(tr), getc(tr)))
                    end
                end
                ix_nb = tr._neighbour_b
                if !visited[ix_nb] #&& !isexternal(t._trigs[ix_nb])
                    cc2=cc
                    nb = t._trigs[ix_nb]
                    cc1=circumcenter(nb)
                    x,y,x1,y1=getx(cc2),gety(cc2),getx(cc1),gety(cc1)
                    if x<min_coord
                        s=(min_coord-x)/(x1-x)
                        cc2=Point2D(min_coord,y+s*(y1-y))
                    elseif x>max_coord
                        s=(max_coord-x)/(x1-x)
                        cc2=Point2D(max_coord,y+s*(y1-y))
                    end
                    x,y=getx(cc2),gety(cc2)
                    if y<min_coord
                        s=(min_coord-y)/(y1-y)
                        cc2=Point2D(x+s*(x1-x),min_coord)
                    elseif y>max_coord
                        s=(max_coord-y)/(y1-y)
                        cc2=Point2D(x+s*(x1-x),max_coord)
                    end
                    x,y=getx(cc2),gety(cc2)
                    if x1<min_coord
                        s=(min_coord-x1)/(x-x1)
                        cc1=Point2D(min_coord,y1+s*(y-y1))
                    elseif x1>max_coord
                        s=(max_coord-x1)/(x-x1)
                        cc1=Point2D(max_coord,y1+s*(y-y1))
                    end
                    x1,y1=getx(cc1),gety(cc1)
                    if y1<min_coord
                        s=(min_coord-y1)/(y-y1)
                        cc1=Point2D(x1+s*(x-x1),min_coord)
                    elseif y1>max_coord
                        s=(max_coord-y1)/(y-y1)
                        cc1=Point2D(x1+s*(x-x1),max_coord)
                    end
                    x,y,x1,y1=getx(cc2),gety(cc2),getx(cc1),gety(cc1)
                    if x != x1 || y != y1 || !isextreme(geta(tr)) || !isextreme(getc(tr))
                        put!(c, VoronoiEdge(cc2, cc1, geta(tr), getc(tr)))
                    end  
                end
                ix_nc = tr._neighbour_c
                if !visited[ix_nc] #&& !isexternal(t._trigs[ix_nb])
                    cc2=cc
                    nb = t._trigs[ix_nc]
                    cc1=circumcenter(nb)
                    x,y,x1,y1=getx(cc2),gety(cc2),getx(cc1),gety(cc1)
                    if x<min_coord
                        s=(min_coord-x)/(x1-x)
                        cc2=Point2D(min_coord,y+s*(y1-y))
                    elseif x>max_coord
                        s=(max_coord-x)/(x1-x)
                        cc2=Point2D(max_coord,y+s*(y1-y))
                    end
                    x,y=getx(cc2),gety(cc2)
                    if y<min_coord
                        s=(min_coord-y)/(y1-y)
                        cc2=Point2D(x+s*(x1-x),min_coord)
                    elseif y>max_coord
                        s=(max_coord-y)/(y1-y)
                        cc2=Point2D(x+s*(x1-x),max_coord)
                    end
                    x,y=getx(cc2),gety(cc2)
                    if x1<min_coord
                        s=(min_coord-x1)/(x-x1)
                        cc1=Point2D(min_coord,y1+s*(y-y1))
                    elseif x1>max_coord
                        s=(max_coord-x1)/(x-x1)
                        cc1=Point2D(max_coord,y1+s*(y-y1))
                    end
                    x1,y1=getx(cc1),gety(cc1)
                    if y1<min_coord
                        s=(min_coord-y1)/(y-y1)
                        cc1=Point2D(x1+s*(x-x1),min_coord)
                    elseif y1>max_coord
                        s=(max_coord-y1)/(y-y1)
                        cc1=Point2D(x1+s*(x-x1),max_coord)
                    end
                    x,y,x1,y1=getx(cc2),gety(cc2),getx(cc1),gety(cc1)
                    if x != x1 || y != y1 || !isextreme(geta(tr)) || !isextreme(getb(tr))
                        put!(c, VoronoiEdge(cc2, cc1, geta(tr), getb(tr)))
                    end
                
                end
            end
        else
            for ix in 2:t._last_trig_index
                visited[ix] && continue
                tr = t._trigs[ix]
                visited[ix] = true
                #isexternal(tr) && continue
                cc = circumcenter(tr)

                ix_na = tr._neighbour_a
                if !visited[ix_na] #&& !isexternal(t._trigs[ix_na])
                    cc2=cc
                    nb = t._trigs[ix_na]
                    cc1=circumcenter(nb)
                    x,y,x1,y1=getx(cc2),gety(cc2),getx(cc1),gety(cc1)
                    if x<min_coord1
                        s=(min_coord1-x)/(x1-x)
                        cc2=Point2D(min_coord1,y+s*(y1-y))
                    elseif x>max_coord1
                        s=(max_coord1-x)/(x1-x)
                        cc2=Point2D(max_coord1,y+s*(y1-y))
                    end
                    x,y=getx(cc2),gety(cc2)
                    if y<min_coord1
                        s=(min_coord1-y)/(y1-y)
                        cc2=Point2D(x+s*(x1-x),min_coord1)
                    elseif y>max_coord1
                        s=(max_coord1-y)/(y1-y)
                        cc2=Point2D(x+s*(x1-x),max_coord1)
                    end
                    x,y=getx(cc2),gety(cc2)
                    if x1<min_coord1
                        s=(min_coord1-x1)/(x-x1)
                        cc1=Point2D(min_coord1,y1+s*(y-y1))
                    elseif x1>max_coord1
                        s=(max_coord1-x1)/(x-x1)
                        cc1=Point2D(max_coord1,y1+s*(y-y1))
                    end
                    x1,y1=getx(cc1),gety(cc1)
                    if y1<min_coord1
                        s=(min_coord1-y1)/(y-y1)
                        cc1=Point2D(x1+s*(x-x1),min_coord1)
                    elseif y1>max_coord1
                        s=(max_coord1-y1)/(y-y1)
                        cc1=Point2D(x1+s*(x-x1),max_coord1)
                    end
                    x,y,x1,y1=getx(cc2),gety(cc2),getx(cc1),gety(cc1)
                    if x != x1 || y != y1 || !isextreme(getb(tr)) || !isextreme(getc(tr))
                        put!(c, VoronoiEdge(cc2, cc1, getb(tr), getc(tr)))
                    end
                end
                ix_nb = tr._neighbour_b
                if !visited[ix_nb] #&& !isexternal(t._trigs[ix_nb])
                    cc2=cc
                    nb = t._trigs[ix_nb]
                    cc1=circumcenter(nb)
                    x,y,x1,y1=getx(cc2),gety(cc2),getx(cc1),gety(cc1)
                    if x<min_coord1
                        s=(min_coord1-x)/(x1-x)
                        cc2=Point2D(min_coord1,y+s*(y1-y))
                    elseif x>max_coord1
                        s=(max_coord1-x)/(x1-x)
                        cc2=Point2D(max_coord1,y+s*(y1-y))
                    end
                    x,y=getx(cc2),gety(cc2)
                    if y<min_coord1
                        s=(min_coord1-y)/(y1-y)
                        cc2=Point2D(x+s*(x1-x),min_coord1)
                    elseif y>max_coord1
                        s=(max_coord1-y)/(y1-y)
                        cc2=Point2D(x+s*(x1-x),max_coord1)
                    end
                    x,y=getx(cc2),gety(cc2)
                    if x1<min_coord1
                        s=(min_coord1-x1)/(x-x1)
                        cc1=Point2D(min_coord1,y1+s*(y-y1))
                    elseif x1>max_coord1
                        s=(max_coord1-x1)/(x-x1)
                        cc1=Point2D(max_coord1,y1+s*(y-y1))
                    end
                    x1,y1=getx(cc1),gety(cc1)
                    if y1<min_coord1
                        s=(min_coord1-y1)/(y-y1)
                        cc1=Point2D(x1+s*(x-x1),min_coord1)
                    elseif y1>max_coord1
                        s=(max_coord1-y1)/(y-y1)
                        cc1=Point2D(x1+s*(x-x1),max_coord1)
                    end
                    x,y,x1,y1=getx(cc2),gety(cc2),getx(cc1),gety(cc1)
                    if x != x1 || y != y1 || !isextreme(geta(tr)) || !isextreme(getc(tr))
                        put!(c, VoronoiEdge(cc2, cc1, geta(tr), getc(tr)))
                    end  
                end
                ix_nc = tr._neighbour_c
                if !visited[ix_nc] #&& !isexternal(t._trigs[ix_nb])
                    cc2=cc
                    nb = t._trigs[ix_nc]
                    cc1=circumcenter(nb)
                    x,y,x1,y1=getx(cc2),gety(cc2),getx(cc1),gety(cc1)
                    if x<min_coord1
                        s=(min_coord1-x)/(x1-x)
                        cc2=Point2D(min_coord1,y+s*(y1-y))
                    elseif x>max_coord1
                        s=(max_coord1-x)/(x1-x)
                        cc2=Point2D(max_coord1,y+s*(y1-y))
                    end
                    x,y=getx(cc2),gety(cc2)
                    if y<min_coord1
                        s=(min_coord1-y)/(y1-y)
                        cc2=Point2D(x+s*(x1-x),min_coord1)
                    elseif y>max_coord1
                        s=(max_coord1-y)/(y1-y)
                        cc2=Point2D(x+s*(x1-x),max_coord1)
                    end
                    x,y=getx(cc2),gety(cc2)
                    if x1<min_coord1
                        s=(min_coord1-x1)/(x-x1)
                        cc1=Point2D(min_coord1,y1+s*(y-y1))
                    elseif x1>max_coord1
                        s=(max_coord1-x1)/(x-x1)
                        cc1=Point2D(max_coord1,y1+s*(y-y1))
                    end
                    x1,y1=getx(cc1),gety(cc1)
                    if y1<min_coord1
                        s=(min_coord1-y1)/(y-y1)
                        cc1=Point2D(x1+s*(x-x1),min_coord1)
                    elseif y1>max_coord1
                        s=(max_coord1-y1)/(y-y1)
                        cc1=Point2D(x1+s*(x-x1),max_coord1)
                    end
                    x,y,x1,y1=getx(cc2),gety(cc2),getx(cc1),gety(cc1)
                    if x != x1 || y != y1 || !isextreme(geta(tr)) || !isextreme(getb(tr))
                        put!(c, VoronoiEdge(cc2, cc1, geta(tr), getb(tr)))
                    end
                
                end
            end
        end
    end
    Channel(voronoiiterator)
end


        

function voronoiedgeswithoutgenerators(t::DelaunayTessellation2D)
    visited = zeros(Bool, t._last_trig_index)
    visited[1] = true
    function voronoiiterator(c::Channel)
        for ix in 2:t._last_trig_index
            visited[ix] && continue
            tr = t._trigs[ix]
            visited[ix] = true
            #isexternal(tr) && continue
            cc = circumcenter(tr)

            ix_na = tr._neighbour_a
            if !visited[ix_na] #&& !isexternal(t._trigs[ix_na])
                nb = t._trigs[ix_na]
                put!(c, VoronoiEdgeWithoutGenerators(cc, circumcenter(nb)))
            end
            ix_nb = tr._neighbour_b
            if !visited[ix_nb] #&& !isexternal(t._trigs[ix_nb])
                nb = t._trigs[ix_nb]
                put!(c, VoronoiEdgeWithoutGenerators(cc, circumcenter(nb)))
            end
            ix_nc = tr._neighbour_c
            if !visited[ix_nc] #&& !isexternal(t._trigs[ix_nc])
                nb = t._trigs[ix_nc]
                put!(c, VoronoiEdgeWithoutGenerators(cc, circumcenter(nb)))
            end
        end
    end
    Channel(voronoiiterator)
end

#The cell is constructed such as cell._summits gives a ordered list of consecutive summits
function voronoicells(t::DelaunayTessellation2D)
    cells=[[] for i in 1:t._total_points_added]
    gens=[]
    for edge in voronoiedges(t)
        a,b=getgena(edge),getgenb(edge)
        i,j=findfirst(x->x==a,gens),findfirst(x->x==b,gens)
        if i==nothing && getx(a)>=min_coord && getx(a)<=max_coord &&  gety(a)>=min_coord && gety(a)<=max_coord
            push!(gens,a)
            i=length(gens)
        end
        if j==nothing && getx(b)>=min_coord && getx(b)<=max_coord &&  gety(b)>=min_coord && gety(b)<=max_coord
            push!(gens,b)
            j=length(gens)
            
        end
        if i != nothing
            push!(cells[i],edge)
        end
        if j != nothing
            push!(cells[j],edge)
        end
    end
    function celliterator(c::Channel)
        mic,mac=0.,0.
        if t._IO=="Inside"
            mic,mac=min_coord1,max_coord1
        else
            mic,mac=min_coord,max_coord
        end
        for i in 1:t._total_points_added
            cell=cells[i]
            gen=gens[i]
            n=length(cell)
            if n>0
                points=[geta(cell[1]),getb(cell[1])]
                tempo=getb(cell[1])
                deleteat!(cell,1)
                n-=1
            end
            while n>0
                xy=[getx(tempo)==mic||getx(tempo)==mac,gety(tempo)==mic||gety(tempo)==mac]
                k,flag=0,false
                while k<n && flag==false && xy==[false,false]
                    k+=1
                    flag=flag||(geta(cell[k])==tempo)||(getb(cell[k])==tempo)
                end
                if flag==true
                    if geta(cell[k])==tempo
                        push!(points,getb(cell[k]))
                        tempo=getb(cell[k])
                    else
                        push!(points,geta(cell[k]))
                        tempo=geta(cell[k])
                    end
                else
                    if xy[1]
                        while k<n && flag==false
                            k+=1
                            flag=flag||(getx(geta(cell[k]))==getx(tempo))||(getx(getb(cell[k]))==getx(tempo))
                        end
                        if getx(geta(cell[k]))==getx(tempo)
                            push!(points,geta(cell[k]),getb(cell[k]))
                            tempo=getb(cell[k])
                        else
                            push!(points,getb(cell[k]),geta(cell[k]))
                            tempo=geta(cell[k])
                        end
                    elseif xy[2]
                        while k<n && flag==false
                            k+=1
                            flag=flag||(gety(geta(cell[k]))==gety(tempo))||(gety(getb(cell[k]))==gety(tempo))
                        end
                        if gety(geta(cell[k]))==gety(tempo)
                            push!(points,geta(cell[k]),getb(cell[k]))
                            tempo=getb(cell[k])
                        else
                            push!(points,getb(cell[k]),geta(cell[k]))
                            tempo=geta(cell[k])
                        end
                    end
                end
                deleteat!(cell,k)
                n-=1
            end
            put!(c,VoronoiCell(points,gen))
        end
    end
    Channel(celliterator)
end



mutable struct TrigIter
    ix::Int64
end

function iterate(t::DelaunayTessellation2D, it::TrigIter=TrigIter(2)) 
    while isexternal(t._trigs[it.ix]) && it.ix <= t._last_trig_index
        it.ix += 1
    end
    if it.ix > t._last_trig_index
        return nothing
    end
    @inbounds trig = t._trigs[it.ix]
    it.ix += 1
    return (trig, it)
end

#finds the index of the triangle in which p lies
function findindex(tess::DelaunayTessellation2D{T}, p::T) where T<:AbstractPoint2D
    i::Int64 = tess._last_trig_index
    while true
        @inbounds w = intriangle(tess._trigs[i], p)
        w > 0 && return i
        @inbounds tr = tess._trigs[i]
        if w == -1
            i = tr._neighbour_a
        elseif w == -2
            i = tr._neighbour_b
        else
            i = tr._neighbour_c
        end
    end
end

locate(t::DelaunayTessellation2D{T}, p::T) where {T<:AbstractPoint2D} = t._trigs[findindex(t, p)]

movea(tess::DelaunayTessellation2D{T},
      trig::DelaunayTriangle{T}) where {T<:AbstractPoint2D} = tess._trigs[trig._neighbour_a]
moveb(tess::DelaunayTessellation2D{T},
      trig::DelaunayTriangle{T}) where {T<:AbstractPoint2D} = tess._trigs[trig._neighbour_b]
movec(tess::DelaunayTessellation2D{T},
      trig::DelaunayTriangle{T}) where {T<:AbstractPoint2D} = tess._trigs[trig._neighbour_c]

function _pushunfixed!(tess::DelaunayTessellation2D{T}, p::T) where T<:AbstractPoint2D
    i = findindex(tess, p)
    ltrigs1::Int64 = tess._last_trig_index+1
    ltrigs2::Int64 = tess._last_trig_index+2

    @inbounds t1 = tess._trigs[i]
    old_t1_a = geta(t1)
    seta(t1, p)
    old_t1_b = t1._neighbour_b
    old_t1_c = t1._neighbour_c
    t1._neighbour_b = ltrigs1
    t1._neighbour_c = ltrigs2

    @inbounds t2 = tess._trigs[ltrigs1]
    setabc(t2, old_t1_a, p, getc(t1))
    t2._neighbour_a = i
    t2._neighbour_b = old_t1_b
    t2._neighbour_c = ltrigs2

    @inbounds t3 = tess._trigs[ltrigs2]
    setabc(t3, old_t1_a, getb(t1), p)
    t3._neighbour_a = i
    t3._neighbour_b = ltrigs1
    t3._neighbour_c = old_t1_c

    @inbounds nt2 = tess._trigs[t2._neighbour_b]
    if nt2._neighbour_a==i
        nt2._neighbour_a = ltrigs1
    elseif nt2._neighbour_b==i
        nt2._neighbour_b = ltrigs1
    else
        nt2._neighbour_c = ltrigs1
    end

    @inbounds nt3 = tess._trigs[t3._neighbour_c]
    if nt3._neighbour_a==i
        nt3._neighbour_a = ltrigs2
    elseif nt3._neighbour_b==i
        nt3._neighbour_b = ltrigs2
    else
        nt3._neighbour_c = ltrigs2
    end

    tess._last_trig_index += 2

    i
end


#flip operations of Springel,2010
function _flipa!(tess::DelaunayTessellation2D, ix1::Int64, ix2::Int64)
    @inbounds ot1 = tess._trigs[ix1]
    @inbounds ot2 = tess._trigs[ix2]
    if ot2._neighbour_a == ix1
        _flipaa!(tess, ix1, ix2, ot1, ot2)
    elseif ot2._neighbour_b == ix1
        _flipab!(tess, ix1, ix2, ot1, ot2)
    else
        _flipac!(tess, ix1, ix2, ot1, ot2)
    end
end

function _endflipa!(tess::DelaunayTessellation2D,
                    ix1::Int64, ix2::Int64,
                    ot1::DelaunayTriangle, ot2::DelaunayTriangle)
    @inbounds n1 = tess._trigs[ot1._neighbour_a]
    if n1._neighbour_a==ix2
        n1._neighbour_a = ix1
    elseif n1._neighbour_b==ix2
        n1._neighbour_b = ix1
    else
        n1._neighbour_c = ix1
    end
    @inbounds n2 = tess._trigs[ot2._neighbour_c]
    if n2._neighbour_a==ix1
        n2._neighbour_a = ix2
    elseif n2._neighbour_b==ix1
        n2._neighbour_b = ix2
    else
        n2._neighbour_c = ix2
    end
end

function _flipaa!(tess::DelaunayTessellation2D{T},
                  ix1::Int64, ix2::Int64,
                  ot1::DelaunayTriangle{T}, ot2::DelaunayTriangle{T}) where T<:AbstractPoint2D
    old_ot1_geom_b = getb(ot1)
    setb(ot1, geta(ot2))
    ot1._neighbour_a = ot2._neighbour_c
    old_ot1_neighbour_c = ot1._neighbour_c
    ot1._neighbour_c = ix2

    newc = geta(ot2)
    setabc(ot2, geta(ot1), old_ot1_geom_b, newc)
    ot2._neighbour_a = ot2._neighbour_b
    ot2._neighbour_b = ix1
    ot2._neighbour_c = old_ot1_neighbour_c

    _endflipa!(tess,ix1,ix2,ot1,ot2)
end

function _flipab!(tess::DelaunayTessellation2D{T},
                  ix1::Int64, ix2::Int64,
                  ot1::DelaunayTriangle{T}, ot2::DelaunayTriangle{T}) where T<:AbstractPoint2D
    old_ot1_geom_b = getb(ot1)
    setb(ot1, getb(ot2))
    ot1._neighbour_a = ot2._neighbour_a
    old_ot1_neighbour_c = ot1._neighbour_c
    ot1._neighbour_c = ix2

    newc = getb(ot2)
    setabc(ot2, geta(ot1), old_ot1_geom_b, newc)
    ot2._neighbour_a = ot2._neighbour_c
    ot2._neighbour_b = ix1
    ot2._neighbour_c = old_ot1_neighbour_c

    _endflipa!(tess,ix1,ix2,ot1,ot2)
end

function _flipac!(tess::DelaunayTessellation2D{T},
                  ix1::Int64, ix2::Int64,
                  ot1::DelaunayTriangle{T}, ot2::DelaunayTriangle{T}) where T<:AbstractPoint2D
    old_ot1_geom_b = getb(ot1)
    setb(ot1, getc(ot2))
    ot1._neighbour_a = ot2._neighbour_b
    old_ot1_neighbour_c = ot1._neighbour_c
    ot1._neighbour_c = ix2

    setab(ot2, geta(ot1), old_ot1_geom_b)
    ot2._neighbour_b = ix1
    ot2._neighbour_c = old_ot1_neighbour_c

    _endflipa!(tess,ix1,ix2,ot1,ot2)
end

########################

function _flipb!(tess::DelaunayTessellation2D{T},
                 ix1::Int64, ix2::Int64) where T<:AbstractPoint2D
    @inbounds ot1 = tess._trigs[ix1]
    @inbounds ot2 = tess._trigs[ix2]
    if ot2._neighbour_a == ix1
        _flipba!(tess, ix1, ix2, ot1, ot2)
    elseif ot2._neighbour_b == ix1
        _flipbb!(tess, ix1, ix2, ot1, ot2)
    else
        _flipbc!(tess, ix1, ix2, ot1, ot2)
    end
end

function _endflipb!(tess::DelaunayTessellation2D{T},
                    ix1::Int64, ix2::Int64,
                    ot1::DelaunayTriangle{T}, ot2::DelaunayTriangle{T}) where T<:AbstractPoint2D
    @inbounds n1 = tess._trigs[ot1._neighbour_b]
    if n1._neighbour_a==ix2
        n1._neighbour_a = ix1
    elseif n1._neighbour_b==ix2
        n1._neighbour_b = ix1
    else
        n1._neighbour_c = ix1
    end
    @inbounds n2 = tess._trigs[ot2._neighbour_a]
    if n2._neighbour_a==ix1
        n2._neighbour_a = ix2
    elseif n2._neighbour_b==ix1
        n2._neighbour_b = ix2
    else
        n2._neighbour_c = ix2
    end
end

function _flipba!(tess::DelaunayTessellation2D{T},
                  ix1::Int64, ix2::Int64,
                  ot1::DelaunayTriangle{T}, ot2::DelaunayTriangle{T}) where T<:AbstractPoint2D
    old_ot1_geom_c = getc(ot1)
    setc(ot1, geta(ot2))
    old_ot1_neighbour_a = ot1._neighbour_a
    ot1._neighbour_a = ix2
    ot1._neighbour_b = ot2._neighbour_c

    setbc(ot2, getb(ot1), old_ot1_geom_c)
    ot2._neighbour_a = old_ot1_neighbour_a
    ot2._neighbour_c = ix1

    _endflipb!(tess,ix1,ix2,ot1,ot2)
end

function _flipbb!(tess::DelaunayTessellation2D{T},
                  ix1::Int64, ix2::Int64,
                  ot1::DelaunayTriangle{T}, ot2::DelaunayTriangle{T}) where T<:AbstractPoint2D
    old_ot1_geom_c = getc(ot1)
    setc(ot1, getb(ot2))
    old_ot1_neighbour_a = ot1._neighbour_a
    ot1._neighbour_a = ix2
    ot1._neighbour_b = ot2._neighbour_a

    newa = getb(ot2)
    setabc(ot2, newa, getb(ot1), old_ot1_geom_c)
    ot2._neighbour_a = old_ot1_neighbour_a
    ot2._neighbour_b = ot2._neighbour_c
    ot2._neighbour_c = ix1

    _endflipb!(tess,ix1,ix2,ot1,ot2)
end

function _flipbc!(tess::DelaunayTessellation2D{T},
                  ix1::Int64, ix2::Int64,
                  ot1::DelaunayTriangle{T}, ot2::DelaunayTriangle{T}) where T<:AbstractPoint2D
    old_ot1_geom_c = getc(ot1)
    setc(ot1, getc(ot2))
    old_ot1_neighbour_a = ot1._neighbour_a
    ot1._neighbour_a = ix2
    ot1._neighbour_b = ot2._neighbour_b

    newa = getc(ot2)
    setabc(ot2, newa, getb(ot1), old_ot1_geom_c)
    ot2._neighbour_b = ot2._neighbour_a
    ot2._neighbour_a = old_ot1_neighbour_a
    ot2._neighbour_c = ix1

    _endflipb!(tess,ix1,ix2,ot1,ot2)
end

########################

function _flipc!(tess::DelaunayTessellation2D{T},
                 ix1::Int64, ix2::Int64) where T<:AbstractPoint2D
    @inbounds ot1 = tess._trigs[ix1]
    @inbounds ot2 = tess._trigs[ix2]
    if ot2._neighbour_a == ix1
        _flipca!(tess, ix1, ix2, ot1, ot2)
    elseif ot2._neighbour_b == ix1
        _flipcb!(tess, ix1, ix2, ot1, ot2)
    else
        _flipcc!(tess, ix1, ix2, ot1, ot2)
    end
end

function _endflipc!(tess::DelaunayTessellation2D{T},
                    ix1::Int64, ix2::Int64,
                    ot1::DelaunayTriangle{T}, ot2::DelaunayTriangle{T}) where T<:AbstractPoint2D
    @inbounds n1 = tess._trigs[ot1._neighbour_c]
    if n1._neighbour_a==ix2
        n1._neighbour_a = ix1
    elseif n1._neighbour_b==ix2
        n1._neighbour_b = ix1
    else
        n1._neighbour_c = ix1
    end
    @inbounds n2 = tess._trigs[ot2._neighbour_b]
    if n2._neighbour_a==ix1
        n2._neighbour_a = ix2
    elseif n2._neighbour_b==ix1
        n2._neighbour_b = ix2
    else
        n2._neighbour_c = ix2
    end
end

function _flipca!(tess::DelaunayTessellation2D{T},
                  ix1::Int64, ix2::Int64,
                  ot1::DelaunayTriangle{T}, ot2::DelaunayTriangle{T}) where T<:AbstractPoint2D
    old_ot1_geom_a = geta(ot1)
    seta(ot1, geta(ot2))
    old_ot1_neighbour_b = ot1._neighbour_b
    ot1._neighbour_b = ix2
    ot1._neighbour_c = ot2._neighbour_c

    newb = geta(ot2)
    setabc(ot2, old_ot1_geom_a, newb, getc(ot1))
    ot2._neighbour_a = ix1
    ot2._neighbour_c = ot2._neighbour_b
    ot2._neighbour_b = old_ot1_neighbour_b

    _endflipc!(tess,ix1,ix2,ot1,ot2)
end

function _flipcb!(tess::DelaunayTessellation2D{T},
                  ix1::Int64, ix2::Int64,
                  ot1::DelaunayTriangle{T}, ot2::DelaunayTriangle{T}) where T<:AbstractPoint2D
    old_ot1_geom_a = geta(ot1)
    seta(ot1, getb(ot2))
    old_ot1_neighbour_b = ot1._neighbour_b
    ot1._neighbour_b = ix2
    ot1._neighbour_c = ot2._neighbour_a

    setac(ot2, old_ot1_geom_a, getc(ot1))
    ot2._neighbour_a = ix1
    ot2._neighbour_b = old_ot1_neighbour_b

    _endflipc!(tess,ix1,ix2,ot1,ot2)
end

function _flipcc!(tess::DelaunayTessellation2D{T},
                  ix1::Int64, ix2::Int64,
                  ot1::DelaunayTriangle{T}, ot2::DelaunayTriangle{T}) where T<:AbstractPoint2D
    old_ot1_geom_a = geta(ot1)
    seta(ot1, getc(ot2))
    old_ot1_neighbour_b = ot1._neighbour_b
    ot1._neighbour_b = ix2
    ot1._neighbour_c = ot2._neighbour_b

    newb = getc(ot2)
    setabc(ot2, old_ot1_geom_a, newb, getc(ot1))
    ot2._neighbour_c = ot2._neighbour_a
    ot2._neighbour_a = ix1
    ot2._neighbour_b = old_ot1_neighbour_b

    _endflipc!(tess,ix1,ix2,ot1,ot2)
end


#restaures the delaunayhood of a tessellation
function _restoredelaunayhood!(tess::DelaunayTessellation2D{T},
                               ix_trig::Int64) where T<:AbstractPoint2D
    @inbounds center_pt = geta(tess._trigs[ix_trig])

    # `A` - edge
    push!(tess._edges_to_check, ix_trig)
    while length(tess._edges_to_check) > 0
        @inbounds trix = tess._edges_to_check[end]
        @inbounds tr_i = tess._trigs[trix]
        @inbounds nb_a = tr_i._neighbour_a
        if nb_a > 1
            @inbounds tr_f = tess._trigs[nb_a]
            if incircle(tr_f, center_pt) > 0
                _flipa!(tess, trix, nb_a)
                push!(tess._edges_to_check, nb_a)
                continue
            end
        end
        pop!(tess._edges_to_check)
    end

    # `B` - edge
    push!(tess._edges_to_check, tess._last_trig_index-1)
    while length(tess._edges_to_check) > 0
        @inbounds trix = tess._edges_to_check[end]
        @inbounds tr_i = tess._trigs[trix]
        @inbounds nb_b = tr_i._neighbour_b
        if nb_b > 1
            @inbounds tr_f = tess._trigs[nb_b]
            if incircle(tr_f, center_pt) > 0
                _flipb!(tess, trix, nb_b)
                push!(tess._edges_to_check, nb_b)
                continue
            end
        end
        pop!(tess._edges_to_check)
    end

    # `C` - edge
    push!(tess._edges_to_check, tess._last_trig_index)
    while length(tess._edges_to_check) > 0
        @inbounds trix = tess._edges_to_check[end]
        @inbounds tr_i = tess._trigs[trix]
        @inbounds nb_c = tr_i._neighbour_c
        if nb_c > 1
            @inbounds tr_f = tess._trigs[nb_c]
            if incircle(tr_f, center_pt) > 0
                _flipc!(tess, trix, nb_c)
                push!(tess._edges_to_check, nb_c)
                continue
            end
        end
        pop!(tess._edges_to_check)
    end
end

function outside(p::T) where T<:AbstractPoint2D
    return getx(p)<min_coord || gety(p)<min_coord || getx(p)>max_coord || gety(p)>max_coord
end
# push a single point. Grows tessellation as needed
function push!(tess::DelaunayTessellation2D{T}, p::T) where T<:AbstractPoint2D
    if outside(p)
        tess._IO="Outside"
    end
    tess._total_points_added += 1
    sizefit_at_least(tess, tess._total_points_added)
    i = _pushunfixed!(tess, p)
    _restoredelaunayhood!(tess, i)
end

# push an array in given order
function _pushunsorted!(tess::DelaunayTessellation2D{T}, a::Array{T, 1}) where T<:AbstractPoint2D
    sizehint!(tess, length(a))
    for p in a
        push!(tess, p)
    end
end

# push an array but sort it first for better performance
function push!(tess::DelaunayTessellation2D{T}, a::Array{T, 1}) where T<:AbstractPoint2D
    m=length(a)
    b=[a[i] for i in 1:m]
    shuffle!(b)
    mssort!(b)
    _pushunsorted!(tess, b)
end


#plots the voronoi/delaunay mesh
function getplotxy(edges)
    edges = collect(edges)
    x = Float64[]
    y = Float64[]
    for e in edges
        push!(x, getx(geta(e)))
        push!(x, getx(getb(e)))
        push!(x, NaN)
        push!(y, gety(geta(e)))
        push!(y, gety(getb(e)))
        push!(y, NaN)
    end
    (x, y)
end

#computes the cells to give a color in the plotting
function getplotcolors(cells)
    cells=collect(cells)
    x = Float64[]
    y = Float64[]
    group=[]
    group_size=[]
    for cell in cells
        a=getgen(cell)
        summits=getsummits(cell)
        m=length(summits)
        k=1
        while k<=m
            summit=summits[k]
            push!(x, getx(summit))
            push!(y, gety(summit))
            k+=1
        end
        push!(group,a)
        push!(group_size,m)
    end
    (x, y, group, group_size)
end

end # module VoronoiDelaunay



