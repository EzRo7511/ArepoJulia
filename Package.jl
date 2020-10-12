

module Package

export parallel_initial_one,parallel_initial_two,
parallel_boucle_one,parallel_boucle_two,parallel_boucle_three,
ghost_IV_init,ghost_IV,ghost_grad_speed,
treebuilding4timestep,computedeltat,deltat_update,mesh_construction,getplot,time_update,drift,
decomposition_periodicbc,decomposition_symmetricbc,computeKtilda,
from_Conservative_to_Intensive,from_Intensive_to_Conservative,Mesh

using Distributed
using Gadfly
using Plots
using RegionTrees
using StaticArrays: SVector
using GeometricalPredicates
using Colors
include("VoronoiDelaunay.jl")

const shape=4 #shape is the number of variables ie for the Euler equation in 2D shape=4, Euler equation 3D shape=5
const gamma=5/3
const Courant=0.4
const thetaC=0.5 #TreePM factor
const kappa=0.5
const nu = 0.3
const khi=1.0
const scale_fact=1. #scale factor to scale the problem

const nprocs=4
const tf=1 #final time
const Dt=tf #Arbitrary time
const mtilda=0.1*scale_fact^2
const Vtilda=0.1*scale_fact^2

const TOL=10^-4

const scale_down=VoronoiDelaunay.scale_down
const width1 = VoronoiDelaunay.width
const min_coord2 = VoronoiDelaunay.min_coord
const max_coord2 = VoronoiDelaunay.max_coord
const marge_min=VoronoiDelaunay.min_coord-VoronoiDelaunay.min_coord1
const marge_max=VoronoiDelaunay.max_coord1-VoronoiDelaunay.max_coord

#I define the type of structure I will store the data in
mutable struct Cell{T<:AbstractPoint2D}
    _GeneratingPoint::T; _ConservativeVariables::Array{Float64,1}; _IntensiveVariables::Array{Float64,1}
    _Volume::Float64; _Borders::Vector{VoronoiDelaunay.VoronoiEdge{T}}; _centreofmass::T ; _perimetre::Float64
    _SoundSpeed::Float64; _Radius::Float64
    _W::Array{Float64,1}
    _averagegradient::Array{Array{Float64,1},1}
    _Deltat::Float64; _K::Int64 ;_Ftot::Array{Float64,1} ;_Type::String ;_process::Int64

    function Cell{T}(GP::T) where T
        new(GP,[0. for i in 1:shape],[0. for i in 1:shape], 0., VoronoiDelaunay.VoronoiEdge{T}[],Point2D(0.,0.), 0.,0., 0., [0.,0.], [[0.,0.],[0.,0.],[0.,0.],[0.,0.]], 0., 0,[0. for i in 1:shape],"",1)
    end
    function Cell{T}(GP::T,CV::Array{Float64,1}, IV::Array{Float64,1},Type::String,process::Int64) where T
        new(GP,CV,IV, 0., VoronoiDelaunay.VoronoiEdge{T}[],Point2D(0.,0.), 0.,0., 0., [0.,0.], [[0.,0.],[0.,0.],[0.,0.],[0.,0.]],0., 0,[0. for i in 1:shape],Type,process)
    end

    # this constructor is good for making copies
    function Cell{T}(GP::T, CV::Array{Float64,1}, IV::Array{Float64,1}, vol::Float64,
                        B::Vector{VoronoiDelaunay.VoronoiEdge{T}}, com::T,p::Float64,
                        SS::Float64, R::Float64, W::Array{Float64,1}, 
                        averagegradient::Array{Array{Float64,1},1},Deltat::Float64, K::Int64, Ftot::Array{Float64,1}, Type::String, process ::Int64) where T
        new(GP,CV,IV,vol,B,com,p,SS,R,W,averagegradient,Deltat,K,Ftot,Type,process)
    end
end
function Cell(GP::T) where T<:AbstractPoint2D
    Cell{T}(GP)
end
function Cell(GP::T,CV::Array{Float64,1}, IV::Array{Float64,1}, Type::String,process::Int64) where T<:AbstractPoint2D
    Cell{T}(GP,CV,IV,Type,process)
end
function Cell(GP::T, CV::Array{Float64,1}, IV::Array{Float64,1}, vol::Float64, 
                        B::Vector{VoronoiDelaunay.VoronoiEdge{T}}, com::T,p::Float64,
                        SS::Float64, R::Float64, W::Array{Float64,1}, 
                        averagegradient::Array{Array{Float64,1},1},Deltat::Float64, K::Int64,Ftot::Array{Float64,1}, Type::String,process::Int64) where T<:AbstractPoint2D
    Cell{T}(GP,CV,IV,vol,B,com,p,SS,R,W,averagegradient,Deltat,K,Ftot,Type,process)
end

function copy(c::Cell{T}) where T<:AbstractPoint2D
    Cell(c._GeneratingPoint, c._ConservativeVariables, c._IntensiveVariables, c._Volume, c._Borders, c._centreofmass,c._perimetre, c._SoundSpeed, c._Radius, c._W, c._averagegradient, c._Deltat, c._K,c._Ftot,c._Type,c._process)
end


# I then define what is a mesh
mutable struct Mesh2D{T<:Point2D}
    _activecells::Vector{Cell{T}}
    _ghostcells::Vector{Cell{T}}
    
    function Mesh2D{T}(cells::Vector{Cell{T}}) where T
        new(cells,Cell{T}[])
    end
    function Mesh2D{T}(activecells::Vector{Cell{T}},ghostcells::Vector{Cell{T}}) where T
        new(activecells,ghostcells)
    end
    function Mesh2D{T}(GP::Vector{T}) where T
        m=length(GP)
        cells=Cell{T}[]
        for i in 1:m
            gp=GP[i]
            push!(cells,Cell{T}(gp))
        end
        new(cells,Cell{T}[])
    end
    function Mesh2D{T}(GP::Vector{T}, CV::Vector{Array{Float64,1}}, IV::Vector{Array{Float64,1}}) where T
        m=length(GP)
        cells=Cell{T}[]
        for i in 1:m
            gp,cv,iv=GP[i],CV[i],IV[i]
            push!(cells,Cell{T}(gp,cv,iv,"active",1))
        end
        new(cells,Cell{T}[])
    end
    function Mesh2D{T}(GP::Vector{T}, CV::Vector{Array{Float64,1}}, IV::Vector{Array{Float64,1}}, vol::Vector{Float64}, 
                        B::Vector{Vector{VoronoiDelaunay.VoronoiEdge{T}}},com::Vector{T},p::Vector{Float64},
                        SS::Vector{Float64}, R::Vector{Float64},
                        W::Vector{Array{Float64,1}},
                        averagegradient::Vector{Vector{Array{Float64,1}}},Deltat::Vector{Float64}, K::Vector{Int64},Ftot::Vector{Array{Float64,1}}) where T
        m=length(GP)
        cells=Cell{T}[]
        for i in 1:m
            gp,cv,iv,Vol,b,Com,P,ss,r,w,AG,deltat,k,ftot=GP[i],CV[i],IV[i],vol[i],B[i],com[i],p[i]SS[i],R[i],W[i],averagegradient[i],Deltat[i],K[i],Ftot[i]
            push!(cells,Cell(gp,cv,iv,Vol,b,Com,ss,r,w,AG,deltat,k,ftot,"active",1))
        end
        new(cells,Cell{T}[])
    end
    function Mesh2D{T}(GPa::Vector{T}, CVa::Vector{Array{Float64,1}}, IVa::Vector{Array{Float64,1}},
                    GPg::Vector{T}, CVg::Vector{Array{Float64,1}}, IVg::Vector{Array{Float64,1}},process::Int64,processg::Vector{Int64}) where T
        m1=length(GPa)
        m2=length(GPg)
        activecells=Cell{T}[]
        ghostcells=Cell{T}[]
        for i in 1:m1
            gp,cv,iv=GPa[i],CVa[i],IVa[i]
            push!(activecells,Cell{T}(gp,cv,iv,"active",process))
        end
        for i in 1:m2
            gp,cv,iv,pg=GPg[i],CVg[i],IVg[i],processg[i]
            push!(ghostcells,Cell{T}(gp,cv,iv,"ghost",pg))
        end
        new(activecells,ghostcells)
    end
    function Mesh2D{T}(GPa::Vector{T}, CVa::Vector{Array{Float64,1}}, IVa::Vector{Array{Float64,1}}, vola::Vector{Float64}, 
                        Ba::Vector{Vector{VoronoiDelaunay.VoronoiEdge{T}}},coma::Vector{T},pa::Vector{Float64},
                        SSa::Vector{Float64}, Ra::Vector{Float64},
                        Wa::Vector{Array{Float64,1}},
                        averagegradienta::Vector{Vector{Array{Float64,1}}},Deltata::Vector{Float64}, Ka::Vector{Int64},Ftota::Vector{Array{Float64,1}},
                        GPg::Vector{T}, CVg::Vector{Array{Float64,1}}, IVg::Vector{Array{Float64,1}}, volg::Vector{Float64}, 
                        Bg::Vector{Vector{VoronoiDelaunay.VoronoiEdge{T}}},comg::Vector{T},pg::Vector{Float64},
                        SSg::Vector{Float64}, Rg::Vector{Float64},
                        Wg::Vector{Array{Float64,1}},
                        averagegradientg::Vector{Vector{Array{Float64,1}}},Deltatg::Vector{Float64}, Kg::Vector{Int64},Ftotg::Vector{Array{Float64,1}},process::Int64,processg::Vector{Int64}) where T
        m1=length(GPa)
        m2=length(GPg)
        activecells=Cell{T}[]
        ghostcells=Cell{T}[]
        for i in 1:m1
            gp,cv,iv,Vol,b,Com,p,ss,r,w,AG,deltat,k,ftot=GPa[i],CVa[i],IVa[i],vola[i],Ba[i],coma[i],pa[i],SSa[i],Ra[i],Wa[i],averagegradienta[i],Deltata[i],Ka[i],Ftota[i]
            push!(activecells,Cell(gp,cv,iv,Vol,b,Com,p,ss,r,w,AG,deltat,k,ftot,"active",process))
        end
        for i in 1:m2
            gp,cv,iv,Vol,b,Com,p,ss,r,w,AG,deltat,k,ftot,Pg=GPg[i],CVg[i],IVg[i],volg[i],Bg[i],comg[i],pg[i],SSg[i],Rg[i],Wg[i],averagegradientg[i],Deltatg[i],Kg[i],Ftotg[i],processg[i]
            push!(ghostcells,Cell(gp,cv,iv,Vol,b,Com,p,ss,r,w,AG,deltat,k,ftot,"ghost",Pg))
        end
        new(activecells,ghostcells)
    end
end

function Mesh2D(cells::Vector{Cell{T}}) where T<:Point2D
    Mesh2D{T}(cells)
end
Mesh(cells)=Mesh2D(cells)
function Mesh2D(activecells::Vector{Cell{T}},ghostcells::Vector{Cell{T}}) where T<:Point2D
    Mesh2D{T}(activecells,ghostcells)
end
Mesh(activecells,ghostcells)=Mesh2D(activecells,ghostcells)

function Mesh2D(GP::Vector{T}) where T<:Point2D
    Mesh2D{T}(GP)
end
Mesh(GP)=Mesh2D(GP)

function Mesh2D(GP::Vector{T},CV::Vector{Array{Float64,1}},IV::Vector{Array{Float64,1}}) where T<:Point2D
    Mesh2D{T}(GP,CV,IV)
end
Mesh(GP,CV,IV)=Mesh2D(GP,CV,IV)

function Mesh2D(GP::Vector{T}, CV::Vector{Array{Float64,1}}, IV::Vector{Array{Float64,1}}, vol::Vector{Float64}, 
                    B::Vector{Vector{VoronoiDelaunay.VoronoiEdge{T}}}, com::Vector{T},p::Vector{Float64},
                    SS::Vector{Float64}, R::Vector{Float64},
                    W::Vector{Array{Float64,1}}, 
                    averagegradient::Vector{Vector{Array{Float64,1}}},Deltat::Vector{Float64}, K::Vector{Int64},Ftot::Vector{Array{Float64,1}}) where T<:Point2D
    Mesh2D{T}(GP,CV,IV,vol,B,com,p,SS,R,W,averagegradient,Deltat,K,Ftot)
end

Mesh(GP,CV,IV,vol,B,com,p,SS,R,W,averagegradient,Deltat,K,Ftot)=Mesh2D(GP,CV,IV,vol,B,com,p,SS,R,W,averagegradient,Deltat,K,Ftot)

function Mesh2D(GPa::Vector{T},CVa::Vector{Array{Float64,1}},IVa::Vector{Array{Float64,1}},GPg::Vector{T},CVg::Vector{Array{Float64,1}},IVg::Vector{Array{Float64,1}},process::Int64,processg::Vector{Int64}) where T<:Point2D
    Mesh2D{T}(GPa,CVa,IVa,GPg,CVg,IVg,process,processg)
end
Mesh(GPa,CVa,IVa,GPg,CVg,IVg,process,processg)=Mesh2D(GPa,CVa,IVa,GPg,CVg,IVg,process,processg)

function Mesh2D(GPa::Vector{T}, CVa::Vector{Array{Float64,1}}, IVa::Vector{Array{Float64,1}}, vola::Vector{Float64}, 
                    Ba::Vector{Vector{VoronoiDelaunay.VoronoiEdge{T}}}, coma::Vector{T},pa::Vector{Float64},
                    SSa::Vector{Float64}, Ra::Vector{Float64},
                    Wa::Vector{Array{Float64,1}}, 
                    averagegradienta::Vector{Vector{Array{Float64,1}}},Deltata::Vector{Float64}, Ka::Vector{Int64},Ftota::Vector{Array{Float64,1}},
                    GPg::Vector{T}, CVg::Vector{Array{Float64,1}}, IVg::Vector{Array{Float64,1}}, volg::Vector{Float64}, 
                    Bg::Vector{Vector{VoronoiDelaunay.VoronoiEdge{T}}}, comg::Vector{T},pg::Vector{Float64},
                    SSg::Vector{Float64}, Rg::Vector{Float64},
                    Wg::Vector{Array{Float64,1}}, 
                    averagegradientg::Vector{Vector{Array{Float64,1}}},Deltatg::Vector{Float64}, Kg::Vector{Int64},Ftotg::Vector{Array{Float64,1}},process::Int64,processg::Vector{Int64}) where T<:Point2D
    Mesh2D{T}(GPa,CVa,IVa,vola,Ba,coma,pa,SSa,Ra,Wa,averagegradienta,Deltata,Ka,Ftota,GPg,CVg,IVg,volg,Bg,comg,pg,SSg,Rg,Wg,averagegradientg,Deltatg,Kg,Ftotg,process,processg)
end

Mesh(GPa,CVa,IVa,vola,Ba,coma,pa,SSa,Ra,Wa,averagegradienta,Deltata,Ka,Ftota,GPg,CVg,IVg,volg,Bg,comg,pg,SSg,Rg,Wg,averagegradientg,Deltatg,Kg,Ftotg,process,processg)=Mesh2D(GPa,CVa,IVa,vola,Ba,coma,pa,SSa,Ra,Wa,averagegradienta,Deltata,Ka,Ftota,GPg,CVg,IVg,volg,Bg,comg,pg,SSg,Rg,Wg,averagegradientg,Deltatg,Kg,Ftotg,process,processg)





#########################################
#    RIEMANN SOLVER AND FLUX LIMITER    #
#########################################

gamma1=(gamma - 1.0) / (2.0 * gamma)
gamma2=(gamma + 1.0) / (2.0 * gamma)
gamma3=2.0 * gamma / (gamma - 1.0)
gamma4=(2.0 / (gamma - 1.0))
gamma5=(2.0 / (gamma + 1.0))
gamma6=(gamma - 1.0) / (gamma + 1.0)
gamma7=0.5 * (gamma - 1.0)
gamma8=1.0 / gamma
gamma9=gamma - 1.0

#Exact
function Riemann(stL,stR)
    
    st_face=[0. for i in 1:shape]
    
    if stL[4]==0. && stR[4]==0.
        st_face=[0.,0.,0.,0.]
        return st_face
    end
    if stL[1]<0.
        println("hoplé gauche")
    elseif stR[1]<0.
        println("hoplé droite")
    end
    
    if stL[1]>0. && stR[1]>0.
        Bool,Press,Vel=riemann(stL,stR)
        if Bool
            st_face=sample_solution_2d(0.0,stL,stR,Press,Vel)
            return st_face
        else
            st_face=sample_solution_vacuum_generate_2d(0.0,stL,stR)
            return st_face
        end
    elseif stL[1]==0. && stR[1]>0.
        st_face=sample_solution_vacuum_left_2d(0.0,stR)
        return st_face
    elseif stR[1]==0. && stL[1]>0.
        st_face=sample_solution_vacuum_right_2d(0.0,stL)
        return st_face
    elseif stR[1]==0. && stL[1]==0.
        st_face=[0.,0.,0.,0.]
        return st_face
    end
end
function riemann(stL,stR)
    MAXITER,Tol=300000,1.0e-8
    FL,FR,FDL,FDR,pold=0.,0.,0.,0.,0.
    cR=sqrt(gamma*stR[4]/stR[1])
    cL=sqrt(gamma*stL[4]/stL[1])
    dVel = stR[2] - stL[2]
    critVel = gamma4 * (cL + cR) - dVel;
    if (critVel < 0)
        return (false,0.,0.)
    end
    p = guess_for_pressure(stL, stR)
    iter = 0
    while (2 * abs((p - pold) / (p + pold)) > Tol && iter < MAXITER)
        pold = p
        FL,FDL=pressure_function(p, stL)
        FR,FDR=pressure_function(p, stR)
        if (iter < MAXITER / 2)
            p -= (FL + FR + dVel) / (FDL + FDR)
        else
            p -= 0.5 * (FL + FR + dVel) / (FDL + FDR)
        end

        if (p < 0.1 * pold)
            p = 0.1 * pold
        end
        FL,FDL=pressure_function(p, stL)
        FR,FDR=pressure_function(p, stR)
    
        if (iter < MAXITER / 2)
            p -= (FL + FR + dVel) / (FDL + FDR)
        else
            p -= 0.5 * (FL + FR + dVel) / (FDL + FDR)
        end
        if (p < 0.1 * pold)
            p = 0.1 * pold
        end
        iter+=1
    end

    Press = p
    Vel   = 0.5 * (stL[2] + stR[2] + FR - FL)
    return (true,Press,Vel)
end
function pressure_function(P,st)
    c=sqrt(gamma*st[4]/st[1])
    if (P <= st[4])
        prat = P / st[4]

        F  = gamma4 * c * ((prat^gamma1) - 1.0)
        FD = (1.0 / (st[1] * c)) * prat^(-gamma2)
        
    else
        ak  = gamma5 / st[1]
        bk  = gamma6 * st[4]
        qrt = sqrt(ak / (bk + P))
        F  = (P - st[4]) * qrt
        FD = (1.0 - 0.5 * (P - st[4]) / (bk + P)) * qrt
    end
    return (F,FD)
end
function guess_for_pressure(stL,stR)
    cR=sqrt(gamma*stR[4]/stR[1])
    cL=sqrt(gamma*stL[4]/stL[1])
    QMAX=2.0
    pmin, pmax=0.,0.
    pv =0.5 * (stL[4] + stR[4]) - 0.125 * (stR[2] - stL[2]) * (stL[1] + stR[1]) * (cL+cR)
    if (stL[4] < stR[4])
        pmin = stL[4]
        pmax = stR[4]
    else
        pmin = stR[4]
        pmax = stL[4]
    end
    if (pmin > 0)
        qrat = pmax / pmin

        if (qrat <= QMAX && (pmin <= pv && pv <= pmax))
            return pv
        else
            if (pv < pmin)
                pnu = (cL+cR) - gamma7 * (stR[2] - stL[2])
                pde = cL / stL[4]^gamma1 + cR / stR[4]^gamma1
                return (pnu / pde)^gamma3
            else #two-shock approximation
                gel = sqrt((gamma5 / stL[1]) / (gamma6 * stL[4] + pv))
                ger = sqrt((gamma5 / stR[1]) / (gamma6 * stR[4] + pv))
                x   = (gel * stL[4] + ger * stR[4] - (stR[2] - stL[2])) / (gel + ger)

                if(x < pmin || x > pmax)
                    x = pmin
                end

                return x
            end
        end
    else
        return (pmin + pmax) / 2
    end
end
function sample_solution_2d(S,stL,stR,Press,Vel)
    c=0.
    cR=sqrt(gamma*stR[4]/stR[1])
    cL=sqrt(gamma*stL[4]/stL[1])
    st=[0. for l in 1:shape]
    if (S <= Vel) #sample point is left of contact
        st[3] = stL[3]
        if (Press <= stL[4]) #left fan
            shl = stL[2] - cL
            if (S <= shl) #left data state
                st[1]   = stL[1]
                st[2]   = stL[2]
                st[4]   = stL[4]
            else
                cml = cL * (Press / stL[4])^gamma1
                stl = Vel - cml
                if (S > stl) #middle left state 
                    st[1]   = stL[1] * (Press / stL[4])^gamma8
                    st[2]  = Vel
                    st[4] = Press
                else #left state inside fan
                    st[2]  = gamma5 * (cL + gamma7 * stL[2] + S)
                    c      = gamma5 * (cL + gamma7 * (stL[2] - S))
                    st[1]  = stL[1] * (c / cL)^gamma4
                    st[4]  = stL[4] * (c / cL)^gamma3
                end
            end          
        else #left shock 
            if (stL[4] > 0)
                pml = Press / stL[4]
                sl  = stL[2] - cL * sqrt(gamma2 * pml + gamma1)
                if (S <= sl) #left data state
                    st[1]   = stL[1]
                    st[2]   = stL[2]
                    st[4]   = stL[4]
                else #middle left state behind shock
                    st[1]   = stL[1] * (pml + gamma6) / (pml * gamma6 + 1.0)
                    st[2]   = Vel
                    st[4]   = Press
                end
            else
                st[1]   = stL[1] / gamma6
                st[2]  = Vel
                st[4] = Press
            end
        end
    else #right of contact
        st[3] = stR[3]
        if (Press <= stR[4]) #right fan
            shr = stR[2] + cR
            if  (S >= shr) #right data state
                st[1]   = stR[1]
                st[2]   = stR[2]
                st[4]   = stR[4]
            else
                cmr = cR * (Press / stR[4])^gamma1
                str = Vel + cmr
                if (S <= str) #middle right state 
                    st[1]   = stR[1] * (Press / stR[4])^gamma8
                    st[2]  = Vel
                    st[4] = Press
                    else #fan right state
                    st[2]  = gamma5 * (-cR + gamma7 * stR[2] + S)
                    c      = gamma5 * (cR - gamma7 * (stR[2] - S))
                    st[1]  = stR[1] * (c / cR)^gamma4
                    st[4]  = stR[4] * (c / cR)^gamma3
                end
            end          
        else #right shock 
            if (stR[4] > 0)
                pmr = Press / stR[4]
                sr  = stR[2] + cR * sqrt(gamma2 * pmr + gamma1)
                if (S >= sr) #right data state
                    st[1]   = stR[1]
                    st[2]   = stR[2]
                    st[4]   = stR[4]
                 else #middle right state behind shock
                    st[1]   = stR[1] * (pmr + gamma6) / (pmr * gamma6 + 1.0)
                    st[2]   = Vel
                    st[4]   = Press
                end
            else
                st[1]   = stR[1] / gamma6
                st[2]  = Vel
                st[4] = Press
            end
        end
    end
    return st
end              
 
function sample_solution_vacuum_generate_2d(S,stL,stR)
    c=0.
    cR=sqrt(gamma*stR[4]/stR[1])
    cL=sqrt(gamma*stL[4]/stL[1])
    st=[0. for l in 1:shape]
    Sl = stL[2] + 2 * cL / (gamma-1)
    Sr = stR[2] - 2 * cR / (gamma-1)
    if (S <= Sl) #left fan
        st[3] = stL[3]
        shl = stL[2] - cL
        if (S <= shl) #left data state
            st[1]   = stL[1]
            st[2]   = stL[2]
            st[4]   = stL[4]
        else #rarefaction fan left state
            st[2]  = gamma5 * (cL + gamma7 * stL[2] + S)
            c      = gamma5 * (cL + gamma7 * (stL[2] - S))
            st[1]  = stL[1] * (c/cL)^gamma4
            st[4]  = stL[4] * (c/cL)^gamma3
        end
    elseif (S >= Sr) #right fan
        shr = stR[2] + cR
        st[3]=stR[3]
        if(S >= shr) #right data state
            st[1]   = stR[1]
            st[2]   = stR[2]
            st[4]   = stR[4]
        else #rarefaction fan right state
            st[2]  = gamma5 * (-cR + gamma7 * stR[2] + S)
            c      = gamma5 * (cR - gamma7 * (stR[2] - S))
            st[1]  = stR[1] * (c/cR)^gamma4
            st[4]  = stR[4] * (c/cR)^gamma3
        end
    else #vacuum in between
        st[2]  = S
        st[1]  = 0.
        st[4]  = 0.

        st[3] = stL[3] + (stR[3] - stL[3]) * (S - Sl) / (Sr - Sl)
    end
    return st
end              
function sample_solution_vacuum_left_2d(S,stR)
    c=0.
    cR=sqrt(gamma*stR[4]/stR[1])
    st=[0. for l in 1:shape]
    Sr = stR[2] - 2 * cR / (gamma-1)
    st[3] = stR[3]
    if (S >= Sr) #right fan
        shr = stR[2] + cR
        if (S >= shr) #right data state
            st[1]   = stR[1]
            st[2]   = stR[2]
            st[4]   = stR[4]
        else #rarefaction fan right state
            st[2]  = gamma5 * (-cR + gamma7 * stR[2] + S)
            c      = gamma5 * (cR - gamma7 * (stR[2] - S))
            st[1]  = stR[1] * (c/cR)^gamma4
            st[4]  = stR[4] * (c/cR)^gamma3
        end
    else #vacuum state
        st[2]=Sr
        st[1]=0.
        st[4]=0.
    end
    return st
end
function sample_solution_vacuum_right_2d(S,stL)
    c=0.
    cL=sqrt(gamma*stL[4]/stL[1])
    st=[0. for l in 1:shape]
    Sl = stL[2] - 2 * cL / (gamma-1)
    st[3] = stL[3]
    if (S <= Sl) #left fan
        shl = stL[2] - cL
        if (S <= shl) #left data state
            st[1]   = stL[1]
            st[2]   = stL[2]
            st[4]   = stL[4]
        else #rarefaction fan left state
            st[2]  = gamma5 * (cL + gamma7 * stL[2] + S)
            c      = gamma5 * (cL + gamma7 * (stL[2] - S))
            st[1]  = stL[1] * (c/cL)^gamma4
            st[4]  = stL[4] * (c/cL)^gamma3
        end
    else #vacuum state
        st[2]=Sl
        st[1]=0.
        st[4]=0.
    end
    return st
end
            
#HLLC
function flux(st)
    flux=[0. for i in 1:shape]
    flux[1]=st[1]*st[2]
    flux[2]=st[1]*st[2]^2+st[4]
    flux[3]=st[1]*st[2]*st[3]
    flux[4]=(st[4]*gamma/(gamma-1)+0.5*(st[2]^2+st[3]^2)*st[1])*st[2]
    return flux
end
    
function hllc_star_fluxes(st,flux,S,S_star)
    Q0 = st[1]*(S-st[2])/(S-S_star)
    Q1 = Q0*S_star
    Q2 = Q0*st[3]
    Q4 = Q0*((st[4]/(gamma-1)+0.5*(st[2]^2+st[3]^2)*st[1])/st[1]+(S_star-st[2])*(S_star +st[3]/(st[1]*(S-st[1]))))
    hllc_flux=[0. for i in 1:shape]
    hllc_flux[1] = flux[1] + S * (Q0 - st[1])
    hllc_flux[2] = flux[2] + S * (Q1 - st[1] * st[2])
    hllc_flux[3] = flux[3] + S * (Q2 - st[1] * st[3])
    hllc_flux[4] = flux[4] + S * (Q4 - (st[4]/(gamma-1)+0.5*(st[2]^2+st[3]^2)*st[1]))
    return (Q0,hllc_flux)
end
    
    
    
    
    
function Riemann_hllc(stL,stR)
    flux=[0. for i in 1:shape]
    st=[0. for i in 1:shape]
    if stL[1] > 0 && stR[1] > 0
        cR=sqrt(gamma*stR[4]/stR[1])
        cL=sqrt(gamma*stL[4]/stL[1])

        SL = min(stL[2] - cL, stR[2] - cR)
        SR = max(stL[2] + cL, stR[2] + cR)

        rho_hat    = 0.5 * (stL[1] + stR[1])
        c_hat   = 0.5 * (cL+cR)
        Press_star = 0.5 * ((stL[4] + stR[4]) + (stL[2] - stR[2]) * (rho_hat * c_hat))
        S_star     = 0.5 * ((stL[2] + stR[2]) + (stL[4] - stR[4]) / (rho_hat * c_hat))

        fluxL=[stL[1]*stL[2],stL[1]*stL[2]^2+stL[4],stL[1]*stL[2]*stL[3],(stL[4]/(gamma-1)+0.5*stL[1]*(stL[2]^2+stL[3]^2)+stR[4])*stR[2]]
        fluxR=[stR[1]*stR[2],stR[1]*stR[2]^2+stR[4],stR[1]*stR[2]*stR[3],(stR[4]/(gamma-1)+0.5*stR[1]*(stR[2]^2+stR[3]^2)+stR[4])*stR[2]]
      
        if SL >= 0.0
            flux=fluxL
            st=stL
        elseif SR <= 0.0
            flux=fluxR
            st=stR
        elseif SL <= 0.0 && S_star >= 0.0
            rho_star,flux = hllc_star_fluxes(stL, fluxL,S_star, SL)
        else
            rho_star,flux = hllc_star_fluxes(stR,fluxR,S_star,SR)
            st=[rho_star, S_star, stL[3],Press_star]
        end
    end
    return (flux,st)
end

#Flux limiter
function flux_limiter(flux,cellL,cellR,dt,interface)
    upwind_mass,upwind_nrg,reduc_fact=0.,0.,1.
    perimetre=0.
    rate=0.9
    stL,stR=cellL._ConservativeVariables,cellR._ConservativeVariables
    if flux[1]>0.
        upwind_mass=stL[1]*cellL._Volume
        perimetre=cellL._perimetre

    elseif flux[1]<0.
        upwind_mass=stR[1]*cellR._Volume
        perimetre=cellR._perimetre
        
    end
    if abs(flux[1]*dt)>rate*upwind_mass
        reduc_fact = rate * upwind_mass  / abs(flux[1] * dt )
    end
    #or
    #if abs(flux[1]*dt*perimetre)>rate*upwind_mass*interface
        #reduc_fact = rate * upwind_mass*interface  / abs(flux[1] * dt * perimetre)
    #end
    return reduc_fact*flux
end







#Function usefull for the geometry

function vector(x::Point2D)
    a,b=getx(x),gety(x)
    return [a,b]
end

function norm2D(a)
    return sqrt(a[1]^2+a[2]^2)
end

function scalarproduct2D(a,b)
    return a[1]*b[1]+a[2]*b[2]
end

#useful for the gradient computation
function PSImaker(a,b,c,d)
    if d>0
        return (a-c)/d
    elseif d<0
        return (b-c)/d
    else
        return 1
    end
end

function Jacobian(State)
    rho,vx,vy,P=State
    Jacobian=[[[0.,0.] for k in 1:shape ] for i in 1:shape]
    for i in 1:shape
        Jacobian[i][i]=[vx,vy]
    end
    Jacobian[1][2],Jacobian[1][3]=[rho,0.],[0.,rho]
    Jacobian[2][4],Jacobian[3][4]=[1/rho,0.],[0.,1/rho]
    Jacobian[4][2],Jacobian[4][3]=[gamma*P,0.],[0.,gamma*P]
    return Jacobian
end

function matrixprod2D(Jacobian,grad)
    result=[0. for i in 1:shape]
    for i in 1:shape
        for k in 1:shape
            result[i]+=scalarproduct2D(Jacobian[i][k],grad[k])
        end
    end
    return result
end

function from_Conservative_to_Intensive(tab)
    intensive=[0. for j in 1:shape]
    intensive[1]=tab[1]
    intensive[2]=tab[2]/tab[1]
    intensive[3]=tab[3]/tab[1]
    intensive[4]=(tab[4]-0.5*(tab[2]^2+tab[3]^2)/tab[1])*(gamma-1)
    return intensive
end
function from_Intensive_to_Conservative(tab)
    conservative=[0. for j in 1:shape]
    conservative[1]=tab[1]
    conservative[2]=tab[2]*tab[1]
    conservative[3]=tab[3]*tab[1]
    conservative[4]=(tab[4]/(gamma-1)+0.5*(tab[2]^2+tab[3]^2)*tab[1])
    return conservative
end



##############


#Finds a point in log(n) in a quadtree that stores data of type Cell
function find(tree, GP::Point2D)
    if GP==Point2D(0.,0.) || GP==Point2D(GeometricalPredicates.min_coord, GeometricalPredicates.min_coord) || GP == Point2D(GeometricalPredicates.max_coord, GeometricalPredicates.min_coord) || GP == Point2D(GeometricalPredicates.min_coord, GeometricalPredicates.max_coord) || GP == Point2D(GeometricalPredicates.max_coord, GeometricalPredicates.max_coord)
        return nothing
    end
    GPT=vector(GP)
    c=tree
    while isleaf(c)==false
        a=c.boundary.origin+c.boundary.widths/2
        bi=[(GPT[1]>a[1])+1,(GPT[2]>a[2])+1]
        c=c[bi[1],bi[2]]
    end
    
    if c.data._GeneratingPoint==GP
        return c
    end
    return nothing
end

#Builds a tree thanks to a given mesh
function build(mesh)
    GeneratingTree=RegionTrees.Cell(SVector(1.0, 1.0), SVector(1-eps(Float64),1-eps(Float64)),Cell(Point2D(0.0,0.0)))
    for cell in mesh._activecells
        Pti=vector(cell._GeneratingPoint)
        c=GeneratingTree
        while isleaf(c)==false
            a=c.boundary.origin+c.boundary.widths/2
            bi=[(Pti[1]>a[1])+1,(Pti[2]>a[2])+1]
            c=c[bi[1],bi[2]]
        end
        if c.data._GeneratingPoint==Point2D(0.0,0.0)
            c.data=cell
        else
            celln=c.data
            c.data=Cell(Point2D(0.0,0.0))
            Ptj=vector(celln._GeneratingPoint)
            split!(c,[Cell(Point2D(0.0,0.0)),Cell(Point2D(0.0,0.0)),Cell(Point2D(0.0,0.0)),Cell(Point2D(0.0,0.0))])
            a=c.boundary.origin+c.boundary.widths/2
            bi=[(Pti[1]>a[1])+1,(Pti[2]>a[2])+1]
            bj=[(Ptj[1]>a[1])+1,(Ptj[2]>a[2])+1]
            while bi==bj
                c=c[bi[1],bi[2]]
                split!(c,[Cell(Point2D(0.0,0.0)),Cell(Point2D(0.0,0.0)),Cell(Point2D(0.0,0.0)),Cell(Point2D(0.0,0.0))])
                a=c.boundary.origin+c.boundary.widths/2
                bi=[(Pti[1]>a[1])+1,(Pti[2]>a[2])+1]
                bj=[(Ptj[1]>a[1])+1,(Ptj[2]>a[2])+1]
            end
            c[bi[1],bi[2]].data=cell
            c[bj[1],bj[2]].data=celln
        end
    end
    for cell in mesh._ghostcells
        Pti=vector(cell._GeneratingPoint)
        c=GeneratingTree
        while isleaf(c)==false
            a=c.boundary.origin+c.boundary.widths/2
            bi=[(Pti[1]>a[1])+1,(Pti[2]>a[2])+1]
            c=c[bi[1],bi[2]]
        end
        if c.data._GeneratingPoint==Point2D(0.0,0.0)
            c.data=cell
        else
            celln=c.data
            c.data=Cell(Point2D(0.0,0.0))
            Ptj=vector(celln._GeneratingPoint)
            split!(c,[Cell(Point2D(0.0,0.0)),Cell(Point2D(0.0,0.0)),Cell(Point2D(0.0,0.0)),Cell(Point2D(0.0,0.0))])
            a=c.boundary.origin+c.boundary.widths/2
            bi=[(Pti[1]>a[1])+1,(Pti[2]>a[2])+1]
            bj=[(Ptj[1]>a[1])+1,(Ptj[2]>a[2])+1]
            while bi==bj
                c=c[bi[1],bi[2]]
                split!(c,[Cell(Point2D(0.0,0.0)),Cell(Point2D(0.0,0.0)),Cell(Point2D(0.0,0.0)),Cell(Point2D(0.0,0.0))])
                a=c.boundary.origin+c.boundary.widths/2
                bi=[(Pti[1]>a[1])+1,(Pti[2]>a[2])+1]
                bj=[(Ptj[1]>a[1])+1,(Ptj[2]>a[2])+1]
            end
            c[bi[1],bi[2]].data=cell
            c[bj[1],bj[2]].data=celln
        end
    end
    return GeneratingTree
end


#Builds the Voronoi mesh associated to a tree and computes : centre of mass of a cell, perimetre and volume
function space_building(tree)
    tess=VoronoiDelaunay.DelaunayTessellation()
    GenPoints=Point2D[]
    for c in allleaves(tree)
        c.data._Borders=VoronoiDelaunay.VoronoiEdge{Point2D}[]
        if c.data._GeneratingPoint != Point2D(0.,0.) && c.data._Type=="active"
            push!(GenPoints,c.data._GeneratingPoint)
            c.data._Volume=0.
            c.data._centreofmass=Point2D(0.,0.)
            c.data._perimetre=0.
        elseif c.data._GeneratingPoint != Point2D(0.,0.) && c.data._Type=="ghost"
            push!(GenPoints,c.data._GeneratingPoint)
        end
    end
    VoronoiDelaunay.push!(tess,GenPoints)
    for edge in VoronoiDelaunay.voronoiedges(tess)
        alpha = VoronoiDelaunay.getgena(edge)
        beta = VoronoiDelaunay.getgenb(edge)
        alpha1 = VoronoiDelaunay.geta(edge)
        beta1 = VoronoiDelaunay.getb(edge)
        normofface = norm2D(vector(alpha1)-vector(beta1))
        height=norm2D(vector(alpha)-vector(beta))/2
        c = find(tree,alpha)
        cn = find(tree,beta)
        
        if c != nothing && cn != nothing 
            ai=height*normofface/2
            aj=height*normofface/2
            push!(c.data._Borders,edge)
            if c.data._Type == "active"
                c.data._Volume+=abs(ai)
                centreofmass=vector(c.data._centreofmass)
                centreofmass+=(vector(alpha)+vector(alpha1)+vector(beta1))*ai/3
                c.data._centreofmass=Point2D(centreofmass[1],centreofmass[2])
                c.data._perimetre+=normofface
            end
            push!(cn.data._Borders,edge)
            if cn.data._Type == "active"
                cn.data._Volume+=abs(aj)
                centreofmassn=vector(cn.data._centreofmass)
                centreofmassn+=(vector(beta)+vector(alpha1)+vector(beta1))*aj/3
                cn.data._centreofmass=Point2D(centreofmassn[1],centreofmassn[2])
                cn.data._perimetre+=normofface
            end
        end
    end
    for c in allleaves(tree)
        if c.data._Type != "ghost" && c.data._GeneratingPoint !=Point2D(0.,0.)
            centreofmass=vector(c.data._centreofmass)
            centreofmass=centreofmass/c.data._Volume
            c.data._centreofmass=Point2D(centreofmass[1],centreofmass[2])
            c.data._Volume=c.data._Volume*scale_fact^2
            c.data._perimetre=c.data._perimetre*scale_fact
        end
    end
    return tree
end


#tells if a leaf is at a corner of a tree
function iscoin(leaf)
    if leaf.boundary.origin == [min_coord2,min_coord2] || leaf.boundary.origin+leaf.boundary.widths == [max_coord2,max_coord2] || leaf.boundary.origin + [0.,leaf.boundary.widths[2]] == [min_coord2,max_coord2] || leaf.boundary.origin + [leaf.boundary.widths[1],0.] == [max_coord2,min_coord2]
        return true
    else
        return false
    end
end

#Finds a point in log(n) in a quadtree with data stored of type [Point2D,Vector{Float64},Int]
function findtime(tree, GP::Point2D)
    if GP==Point2D(0.,0.) || GP==Point2D(GeometricalPredicates.min_coord, GeometricalPredicates.min_coord) || GP == Point2D(GeometricalPredicates.max_coord, GeometricalPredicates.min_coord) || GP == Point2D(GeometricalPredicates.min_coord, GeometricalPredicates.max_coord) || GP == Point2D(GeometricalPredicates.max_coord, GeometricalPredicates.max_coord)
        return nothing
    end
    GPT=vector(GP)
    c=tree
    while isleaf(c)==false
        a=c.boundary.origin+c.boundary.widths/2
        bi=[(GPT[1]>a[1])+1,(GPT[2]>a[2])+1]
        c=c[bi[1],bi[2]]
    end
    
    if c.data[1]==GP
        return c
    end
    return nothing
end

#builds the tree used for the TreePM method used to compute the speeds of the generating points  
function computeKtilda(mesh)
    sum=0.
    for cell in mesh._activecells
        sum+=1/(cell._ConservativeVariables[1]/mtilda + 1/Vtilda)
    end
    return (scale_fact*scale_down)^2/sum
end

function treebuilding4dis(Tree4dis,tree,Ktilda)
    if isleaf(tree)==true && tree.data._GeneratingPoint != Point2D(0.0,0.0)
        cell=tree.data
        mass=Ktilda/(1/Vtilda+cell._IntensiveVariables[1]/mtilda)
        Position=(vector(cell._GeneratingPoint)*mass-(Tree4dis.boundary.origin+Tree4dis.boundary.widths/2)*Tree4dis.boundary.widths[1]*Tree4dis.boundary.widths[2]*scale_fact^2)/(mass-Tree4dis.boundary.widths[1]*Tree4dis.boundary.widths[2]*scale_fact^2)
        Tree4dis.data=[mass-scale_fact^2*Tree4dis.boundary.widths[1]*Tree4dis.boundary.widths[2],Position,cell._GeneratingPoint]
    elseif isleaf(tree)==true && tree.data._GeneratingPoint==Point2D(0.0,0.0) && !iscoin(tree)
        Tree4dis.data=[-scale_fact^2*Tree4dis.boundary.widths[1]*Tree4dis.boundary.widths[2],Tree4dis.boundary.origin+Tree4dis.boundary.widths/2,Point2D(0.,0.)]
    elseif isleaf(tree)==false
        split!(Tree4dis,[[0,[0,0],Point2D(0.,0.)],[0,[0,0],Point2D(0.,0.)],[0,[0,0],Point2D(0.,0.)],[0,[0,0],Point2D(0.,0.)]])
        treebuilding4dis(Tree4dis[1,1],tree[1,1],Ktilda)
        treebuilding4dis(Tree4dis[1,2],tree[1,2],Ktilda)
        treebuilding4dis(Tree4dis[2,1],tree[2,1],Ktilda)
        treebuilding4dis(Tree4dis[2,2],tree[2,2],Ktilda)
        mass11,mass12,mass21,mass22=Tree4dis[1,1].data[1],Tree4dis[1,2].data[1],Tree4dis[2,1].data[1],Tree4dis[2,2].data[1]
        p11,p12,p21,p22=Tree4dis[1,1].data[2],Tree4dis[1,2].data[2],Tree4dis[2,1].data[2],Tree4dis[2,2].data[2]
        mass=mass11+mass12+mass21+mass22
        p=(mass11*p11+mass12*p12+mass21*p21+mass22*p22)/mass
        Tree4dis.data=[mass,p,Point2D(0.,0.)]
    end
end

#Computes the force applied by a node over a leaf to compute the speed in the Poisson's equation resolution
function force_over(leaf,cell)
    r=scale_fact*(leaf.data[2]-cell.data[2])
    normr=norm2D(r)
    if isleaf(cell)
        if normr==0
            return [0.0,0.0]
        else
            return (-cell.data[1]*r/(4*pi*normr^3))
        end
    else
        flag=true
        flag1=false
        for child in children(cell)
            rc=scale_fact*(leaf.data[2]-child.data[2])
            normrc=norm2D(rc)
            if normrc==0
                flag1=true
            else
                thetac = scale_fact*child.boundary.widths[1]/normrc
                flag = (flag && thetac > thetaC)
            end
        end
        if (flag||flag1)==false
            return (-cell.data[1]*r/(4*pi*normr^3))
        else
            force=[0.0,0.0]
            for child in children(cell)
                force+=force_over(leaf,child)
            end
            return force
        end
    end
end

#Builds the tree used to compute the individual timestep
function treebuilding4timestep(Tree4time,tree)
    if isleaf(tree) && isleaf(Tree4time) && tree.data._GeneratingPoint != Point2D(0.0,0.0) && tree.data._Type=="active"
        cell=tree.data
        GPi,ci,vi,pi=cell._GeneratingPoint,cell._SoundSpeed,norm2D(cell._IntensiveVariables[2:3]),cell._process
        if Tree4time.data[1]==Point2D(0.,0.)
            Tree4time.data=[GPi,[ci,vi],pi]
        else
            GPj=Tree4time.data[1]
            Pti=vector(GPi)
            Ptj=vector(GPj)
            cj,vj=Tree4time.data[2]
            c,v=max(ci,cj),max(vi,vj)
            Tree4time.data=[Point2D(0.,0.),[c,v],1]
            a=Tree4time.boundary.origin+Tree4time.boundary.widths/2
            bi=[(Pti[1]>a[1])+1,(Pti[2]>a[2])+1]
            bj=[(Ptj[1]>a[1])+1,(Ptj[2]>a[2])+1]
            while bi==bj
                Tree4time=Tree4time[bi[1],bi[2]]
                Tree4time.data=[Point2D(0.,0.),[c,v],1]
                split!(Tree4time,[[Point2D(0.,0.),[0.,0.],1],[Point2D(0.,0.),[0.,0.],1],[Point2D(0.,0.),[0.,0.],1],[Point2D(0.,0.),[0.,0.]],1])
                a=Tree4time.boundary.origin+Tree4time.boundary.widths/2
                bi=[(Pti[1]>a[1])+1,(Pti[2]>a[2])+1]
                bj=[(Ptj[1]>a[1])+1,(Ptj[2]>a[2])+1]
            end
            Tree4time[bi[1],bi[2]].data=[GPi,[ci,vi],pi]
            Tree4time[bj[1],bj[2]].data=[GPj,[cj,vj],pj]
        end
    elseif isleaf(tree) && !isleaf(Tree4time) && tree.data._GeneratingPoint != Point2D(0.0,0.0) && tree.data._Type=="active"
        cell=tree.data
        Tree4time.data[2]=[max(cell._SoundSpeed,Tree4time.data[2][1]),max(norm2D(cell._IntensiveVariables[2:3]),Tree4time.data[2][2])]
        Pt=vector(cell.data._GeneratingPoint)
        a=Tree4time.boundary.origin+Tree4time.boundary.widths/2
        b=[(Pt[1]>a[1])+1,(Pt[2]>a[2])+1]
        treebuilding4timestep(Tree4time[b[1],b[2]],tree)
    elseif !isleaf(tree) && isleaf(Tree4time)
        split!(Tree4time,[[Point2D(0.,0.),[0.,0.],1],[Point2D(0.,0.),[0.,0.],1],[Point2D(0.,0.),[0.,0.],1],[Point2D(0.,0.),[0.,0.]],1])
        treebuilding4timestep(Tree4time[1,1],tree[1,1])
        treebuilding4timestep(Tree4time[1,2],tree[1,2])
        treebuilding4timestep(Tree4time[2,1],tree[2,1])
        treebuilding4timestep(Tree4time[2,2],tree[2,2])
        c1,v1=Tree4time[1,1].data[2]
        c2,v2=Tree4time[2,1].data[2]
        c3,v3=Tree4time[1,2].data[2]
        c4,v4=Tree4time[2,2].data[2]
        cmax=max(c1,c2,c3,c4)
        vmax=max(v1,v2,v3,v4)
        Tree4time.data=[Point2D(0.,0.),[cmax,vmax],1]
    elseif !isleaf(tree) && !isleaf(Tree4time)
        treebuilding4timestep(Tree4time[1,1],tree[1,1])
        treebuilding4timestep(Tree4time[1,2],tree[1,2])
        treebuilding4timestep(Tree4time[2,1],tree[2,1])
        treebuilding4timestep(Tree4time[2,2],tree[2,2])
        c1,v1=Tree4time[1,1].data[2]
        c2,v2=Tree4time[2,1].data[2]
        c3,v3=Tree4time[1,2].data[2]
        c4,v4=Tree4time[2,2].data[2]
        cmax=max(c1,c2,c3,c4)
        vmax=max(v1,v2,v3,v4)
        Tree4time.data=[Point2D(0.,0.),[cmax,vmax],1]
    end
end

#Computes the individual timestep of a given active leaf
function treewalk4timestep(leaf,Tree4time,r,deltatcurrent)
    GP=leaf.data[1]
    proc=leaf.data[3]
    c=find(fetch(r[proc-1]),GP)
    vi=[c.data._IntensiveVariables[2],c.data._IntensiveVariables[3]]
    if isleaf(Tree4time) && Tree4time.data[1] != Point2D(0.,0.) && Tree4time.data[1] != GP
        GPn=Tree4time.data[1]
        procn=Tree4time.data[3]
        cn=find(fetch(r[procn-1]),GPn)
        vj=[cn.data._IntensiveVariables[2],cn.data._IntensiveVariables[3]]
        rij=(vector(GP)-vector(GPn))*scale_fact
        vij=vi-vj
        normrij=norm2D(rij)
        vijsign=abs(Tree4time.data[2][1]+leaf.data[2][1]-scalarproduct2D(vij,rij)/normrij)
        if vijsign != 0. && normrij/vijsign < deltatcurrent
            deltatcurrent=normrij/vijsign
        end
        return deltatcurrent
    elseif isleaf(Tree4time) && (Tree4time.data[1] == Point2D(0.,0.)||Tree4time.data[1] == GP)
        return deltatcurrent
    elseif !isleaf(Tree4time) && getx(GP)>Tree4time.boundary.origin[1] && gety(GP)>Tree4time.boundary.origin[2] && getx(GP)<Tree4time.boundary.origin[1]+Tree4time.boundary.widths[1] && gety(GP)<Tree4time.boundary.origin[2]+Tree4time.boundary.widths[2]
        return min(treewalk4timestep(leaf,Tree4time[1,1],r,deltatcurrent),treewalk4timestep(leaf,Tree4time[1,2],r,deltatcurrent),treewalk4timestep(leaf,Tree4time[2,1],r,deltatcurrent),treewalk4timestep(leaf,Tree4time[2,2],r,deltatcurrent))
    else
        dx=0.
        dy=0.
        if getx(GP)<Tree4time.boundary.origin[1]
            dx=Tree4time.boundary.origin[1]-getx(GP)
        elseif getx(GP)>Tree4time.boundary.origin[1]+Tree4time.boundary.widths[1]
            dx=getx(GP)-(Tree4time.boundary.origin[1]+Tree4time.boundary.widths[1])
        end
        if gety(GP)<Tree4time.boundary.origin[2]
            dy=Tree4time.boundary.origin[2]-gety(GP)
        elseif gety(GP)>Tree4time.boundary.origin[2]+Tree4time.boundary.widths[2]
            dy=gety(GP)-(Tree4time.boundary.origin[2]+Tree4time.boundary.widths[2])
        end
        dmin=sqrt(dx^2+dy^2)
        if dmin<deltatcurrent*(Tree4time.data[2][1]+leaf.data[2][1]+Tree4time.data[2][2]+leaf.data[2][2])
            deltatcurrent=min(treewalk4timestep(leaf,Tree4time[1,1],r,deltatcurrent),treewalk4timestep(leaf,Tree4time[1,2],r,deltatcurrent),treewalk4timestep(leaf,Tree4time[2,1],r,deltatcurrent),treewalk4timestep(leaf,Tree4time[2,2],r,deltatcurrent))
        end
        return deltatcurrent
    end
end


#Computes the timestep of every cell
function computedeltat(tree,Tree4time,r)
    for c in allleaves(tree)
        if c.data._GeneratingPoint==Point2D(0.,0.) || c.data._Type=="ghost"
            continue
        end
        leaf=findtime(Tree4time,c.data._GeneratingPoint)
        if leaf==nothing
            continue
        end
        deltatcurrent=c.data._Radius/(leaf.data[2][1]+norm2D(c.data._W-c.data._IntensiveVariables[2:3]))
        c.data._Deltat=Courant*treewalk4timestep(leaf,Tree4time,r,deltatcurrent)
        c.data._K=floor(Int,log(2,Dt/c.data._Deltat)+1)
    end
    return tree
end





#Computes the soundspeed and radius of an active cell
function computeSSRadius(tree)
    for c in allleaves(tree)
        if c.data._GeneratingPoint != Point2D(0.,0.) && c.data._Type == "active"
            if c.data._IntensiveVariables[1]==0.
                c.data._SoundSpeed=0.
            else
                c.data._SoundSpeed=sqrt(c.data._IntensiveVariables[4]*gamma/c.data._IntensiveVariables[1])
            end
            c.data._Radius=sqrt(c.data._Volume/pi)
        end
    end
    return tree
end


#Compute the speed of evaery generating points
function computespeed(tree,Ktilda)
    Tree4dis=RegionTrees.Cell(SVector(min_coord2,min_coord2), SVector(max_coord2,max_coord2),[0,[0,0],Point2D(0.,0.)])
    treebuilding4dis(Tree4dis,tree,Ktilda)
    for leaf in allleaves(Tree4dis)
        Displacement=[0.,0.]
        Position=leaf.data[3]
        c=find(tree,Position)
        if c == nothing || c.data._Type == "ghost"
            continue
        end
        #Displacement+=force_over(leaf,Tree4dis)
        c.data._W=c.data._IntensiveVariables[2:3]
        #delta=c.data._Deltat
        #if delta==0
            #c.data._W-=kappa*Displacement/Dt
        #else
            #c.data._W-=kappa*Displacement/delta
        #end
        #c.data._W-=kappa*Displacement/Dt
        Displacement1=[0.,0.]
        di=(vector(c.data._centreofmass)-vector(c.data._GeneratingPoint))*scale_fact
        normofdi=norm2D(di)
        Ri=c.data._Radius
        ci=c.data._SoundSpeed
        if normofdi/(nu*Ri)<0.9
            Displacement1=[0.,0.]
        elseif  normofdi/(nu*Ri)<1.1
            Displacement1=khi*ci*(normofdi-0.9*nu*Ri)*di/(normofdi*0.2*nu*Ri)
        else
            Displacement1=khi*ci*di/normofdi
        end
        c.data._W+=Displacement1
    end
    return tree
end


#Computes the average gradient in a given cell
function gradientmoyen(tree,GP)
    grad=[[0.,0.] for l in 1:shape]
    ri=GP
    c=find(tree,GP)
    Vi=c.data._Volume
    for border in c.data._Borders
        if GP==VoronoiDelaunay.getgena(border)
            GPn=VoronoiDelaunay.getgenb(border)
        else
            GPn=VoronoiDelaunay.getgena(border)
        end
        cn=find(tree,GPn)
        a,b=VoronoiDelaunay.geta(border), VoronoiDelaunay.getb(border)
        face = (vector(a)-vector(b))*scale_fact
        centreofface = 0.5*(vector(a)+vector(b))*scale_fact
        normofface=norm2D(face)
        rj=GPn
        middleofneighbours=0.5*(vector(ri)+vector(rj))*scale_fact
        rij=(vector(ri)-vector(rj))*scale_fact
        cij=(centreofface-middleofneighbours)
        normofrij=norm2D(rij)
        for l in 1:shape
            grad[l]+=((cn.data._IntensiveVariables[l]-c.data._IntensiveVariables[l])*cij-(cn.data._IntensiveVariables[l]+c.data._IntensiveVariables[l])*rij/2)*normofface/normofrij
        end
    end
    return grad/Vi
end

#Computes and slope limits the gradient of every cell
function computegradients(tree)
    for c in allleaves(tree)
        GP=c.data._GeneratingPoint
        if GP== Point2D(0.0,0.0) || c.data._Type == "ghost"
            continue
        end
        c.data._averagegradient=gradientmoyen(tree,GP)
        
        DELTAij=[[] for l in 1:shape]
        PSIij=[[] for l in 1:shape]
        si=vector(c.data._centreofmass)*scale_fact
        intvar=[c.data._IntensiveVariables[l] for l in 1:shape]
        psimintempo=[intvar[l] for l in 1:shape]
        psimaxtempo=[intvar[l] for l in 1:shape]
        for border in c.data._Borders
            if GP==VoronoiDelaunay.getgena(border)
                GPn=VoronoiDelaunay.getgenb(border)
            else
                GPn=VoronoiDelaunay.getgena(border)
            end
            cn=find(tree,GPn)
            a,b=VoronoiDelaunay.geta(border), VoronoiDelaunay.getb(border)
            centreofface = 0.5*(vector(a)+vector(b))*scale_fact
            psijtempo=cn.data._IntensiveVariables
            for l in 1:shape
                if psijtempo[l]>psimaxtempo[l]
                    psimaxtempo[l]=psijtempo[l]
                end
                if psijtempo[l]<psimintempo[l]
                    psimintempo[l]=psijtempo[l]
                end
                push!(DELTAij[l],scalarproduct2D(c.data._averagegradient[l],centreofface-si))
            end
        end
        for l in 1:shape
            for delta in DELTAij[l]
                push!(PSIij[l],PSImaker(psimaxtempo[l],psimintempo[l],intvar[l],delta))
            end
            alphai=1
            for psi in PSIij[l]
                if psi < alphai
                    alphai=psi
                end
            end
            c.data._averagegradient[l]=alphai*c.data._averagegradient[l]
        end
    end
    return tree
end


#updates the individual timestep of every active cell
function deltat_update(tree,Tree4time,r,currentK::Int64)
    for c in allleaves(tree)
        if c.data._GeneratingPoint==Point2D(0.,0.) || c.data._Type=="ghost" || c.data._K<currentK
            continue
        end
        leaf=findtime(Tree4time,c.data._GeneratingPoint)
        if leaf==nothing
            continue
        end
        if (leaf.data[2][1]+norm2D(c.data._W-c.data._IntensiveVariables[2:3]))!=0. && c.data._Radius != 0.
            deltatcurrent=c.data._Radius/(leaf.data[2][1]+norm2D(c.data._W-c.data._IntensiveVariables[2:3]))
        else
            deltatcurrent=Dt/4
        end
        c.data._Deltat=Courant*treewalk4timestep(leaf,Tree4time,r,deltatcurrent)
        c.data._K=max(currentK,floor(Int,log(2,Dt/c.data._Deltat)+1))
    end
    return tree
end

#updates the flux of every cell in contact with an active interface
function flux_update(tree,currentK::Int64)
    visited=[]
    for c in allleaves(tree)
        if c.data._GeneratingPoint==Point2D(0.,0.) || c.data._K < currentK
            continue
        elseif findfirst(x->x==c.data._GeneratingPoint,visited) == nothing
            push!(visited,c.data._GeneratingPoint)
        end
        GP=c.data._GeneratingPoint
        K=c.data._K
        for border in c.data._Borders
            if GP==VoronoiDelaunay.getgena(border)
                GPn=VoronoiDelaunay.getgenb(border)
            else
                GPn=VoronoiDelaunay.getgena(border)
            end
            if findfirst(x->x==GPn,visited)!=nothing
                continue
            end
            cn=find(tree,GPn)
            if cn.data._Type == "ghost" && c.data._Type == "ghost"
                continue
            end
            if cn.data._K>K
                ascendant,descendant=cn,c
                deltatact=Dt/2.0^cn.data._K
            else
                ascendant,descendant=c,cn
                deltatact=Dt/2.0^K
            
            end
            ri,rj=ascendant.data._GeneratingPoint,descendant.data._GeneratingPoint
            si,sj=vector(ascendant.data._centreofmass),vector(descendant.data._centreofmass)
            StateL,StateR=[ascendant.data._IntensiveVariables[l] for l in 1:shape],[descendant.data._IntensiveVariables[l] for l in 1:shape]
            wi,wj=ascendant.data._W,descendant.data._W
            drWL,drWR=ascendant.data._averagegradient,descendant.data._averagegradient
        
            a,b = VoronoiDelaunay.geta(border), VoronoiDelaunay.getb(border)
            centreofface=0.5*(vector(a)+vector(b))*scale_fact
            rij=(vector(rj)-vector(ri))*scale_fact
            centreij=scale_fact*(vector(ri)+vector(rj))/2
            normrij=norm2D(rij)
            rotcos=rij[1]/normrij
            rotsin=rij[2]/normrij
            
            wprime=scalarproduct2D(wi-wj,centreofface-centreij)*rij/normrij^2
            w=(wi+wj)/2 + wprime
            #If exact
            StateR[2:3]-=w
            StateL[2:3]-=w
            
            #If HLLC
            #wx,wy=w[1],w[2]
            #wt=[wx*rotcos+wy*rotsin,wy*rotcos-wx*rotsin]
    
            maxrho,minrho,maxP,minP=max(StateL[1],StateR[1]),min(StateL[1],StateR[1]),max(StateL[4],StateR[4]),min(StateL[4],StateR[4])
            JL=Jacobian(StateL)
            JR=Jacobian(StateR)
            StateL-=matrixprod2D(JL,drWL)*deltatact/2
            StateR-=matrixprod2D(JR,drWR)*deltatact/2
            for l in 1:shape
                StateL[l]+=scalarproduct2D(drWL[l],centreofface-si) #Ou alors ri
                StateR[l]+=scalarproduct2D(drWR[l],centreofface-sj) #Ou alors rj
            end
            
            #To avoid negativity
            if StateL[4]<0#abs(StateL[4])<TOL && StateL[4]<0
                StateL[4]=0
            end
            if StateR[4]<0#abs(StateR[4])<TOL && StateR[4]<0
                StateR[4]=0
            end
            if StateL[1]<0#abs(StateL[1])<TOL && StateL[1]<0
                StateL[1]=0
            end
            if StateR[1]<0#abs(StateR[1])<TOL && StateR[1]<0
                StateR[1]=0
            end
            
            # If normal
            vxL,vyL,vxR,vyR=StateL[2],StateL[3],StateR[2],StateR[3]
            StateL[2]=vxL*rotcos+vyL*rotsin
            StateL[3]=vyL*rotcos-vxL*rotsin
            StateR[2]=vxR*rotcos+vyR*rotsin
            StateR[3]=vyR*rotcos-vxR*rotsin
            
            State=Riemann(StateL,StateR)
            
            #If HLLC
            #Flux,State=Riemann_hllc(StateL,StateR)
            
            #Fij=[Flux[1],Flux[2]+wt[1]*Flux[1],Flux[3]+wt[2]*Flux[1],Flux[4]+wt[1]*Flux[2]+wt[2]*Flux[3]+0.5*Flux[1]*(wt[1]^2+wt[2]^2)]
            #momx,momy=Fij[2],Fij[3]
            #Fij[2]=momx*rotcos-momy*rotsin
            #Fij[3]=momy*rotcos+momx*rotsin
            
            
            vx,vy=State[2],State[3]
            State[2]=vx*rotcos-vy*rotsin
            State[3]=vy*rotcos+vx*rotsin
            State[2:3]+=w
            
            l=(vector(a)-vector(b))*scale_fact
            Aij=(sqrt(l[1]^2+l[2]^2))*rij/normrij
            #If normal
            Fij=[0. for k in 1:shape]
            Fij[1]=scalarproduct2D(Aij,State[1]*(State[2:3]-w))
            Fij[2:3]=(scalarproduct2D(State[2:3],(State[2:3]-w))*State[1]+State[4])*Aij
            Fij[4]=scalarproduct2D((State[1]*(scalarproduct2D(State[2:3],State[2:3]/2))+State[4]/(gamma-1))*(State[2:3]-w)+State[4]*State[2:3],Aij)

            Pij=flux_limiter(Fij,ascendant.data,descendant.data,deltatact,norm2D(l))
            Pji=-Pij
            
            if isnan(Pij[4])
                println(descendant.data)
                println(ascendant.data)
            end
            
            if ascendant.data._Type == "active"
                ascendant.data._Ftot-=Pij*deltatact
            end
            if descendant.data._Type == "active"
                descendant.data._Ftot-=Pji*deltatact
            end
        end
    end
    return tree
end





#Derefines the mesh
function derefinement(tree)
    deref=[]
    l=length(active)
    for c in allleaves(tree)
        if c.data._GeneratingPoint==Point2D(0.0,0.0)
            continue
        end
        flag = true
        Vi=c.data._Volume
        mi=c.data._ConservativeVariables[1]*Vi
        GP=c.data._GeneratingPoint
        for edge in c.data._Borders
            if findfirst(x->x==VoronoiDelaunay.getgena(edge),deref) != nothing || findfirst(x->x==VoronoiDelaunay.getgenb(edge),deref) != nothing
                flag = false
            end
        end
        if (mi<0.3*mtilda || Vi<0.1*Vtilda) && flag==true
            push!(deref,GP)
        end
    end

    for GP in deref
        bonus_point=Point2D(0.,0.)
        c=find(tree,GP)

        GenPoints1=Point2D[]
        GenPoints2=Point2D[]

        for edge in c.data._Borders
            if GP==VoronoiDelaunay.getgena(edge)
                GPn=VoronoiDelaunay.getgenb(edge)
            else
                GPn=VoronoiDelaunay.getgena(edge)
            end
            if GPn == Point2D(GeometricalPredicates.min_coord, GeometricalPredicates.min_coord) || GPn == Point2D(GeometricalPredicates.max_coord, GeometricalPredicates.min_coord) || GPn == Point2D(GeometricalPredicates.min_coord, GeometricalPredicates.max_coord) || GPn == Point2D(GeometricalPredicates.max_coord, GeometricalPredicates.max_coord)
                bonus_point=GPn
                continue
            end
            push!(GenPoints1,GPn)
            push!(GenPoints2,GPn)
        end

        tess1=VoronoiDelaunay.DelaunayTessellation()
        tess2=VoronoiDelaunay.DelaunayTessellation()
        VoronoiDelaunay.push!(tess1,GenPoints1)
        VoronoiDelaunay.push!(tess2,GenPoints2)
        VoronoiDelaunay.push!(tess1,GP)
        if bonus_point != Point2D(0.,0.)
            push!(GenPoints1,bonus_point)
            push!(GenPoints2,bonus_point)
        end
        push!(GenPoints1,GP)
        m1=length(GenPoints1)
        m2=m1-1
        Volume1 = zeros(m1)
        Volume2 = zeros(m2)
        for edge in VoronoiDelaunay.voronoiedges(tess1)
            alpha = VoronoiDelaunay.getgena(edge)
            beta = VoronoiDelaunay.getgenb(edge)
            alpha1 = VoronoiDelaunay.geta(edge)
            beta1 = VoronoiDelaunay.getb(edge)
            j = findfirst(x->x==alpha,GenPoints1)
            k = findfirst(x->x==beta,GenPoints1)
            if j != nothing && k != nothing
                Volume1[j]+=abs(area(Primitive(alpha,alpha1,beta1)))
                Volume1[k]+=abs(area(Primitive(beta,alpha1,beta1)))
            end
        end
        for edge in VoronoiDelaunay.voronoiedges(tess2)
            alpha = VoronoiDelaunay.getgena(edge)
            beta = VoronoiDelaunay.getgenb(edge)
            alpha1 = VoronoiDelaunay.geta(edge)
            beta1 = VoronoiDelaunay.getb(edge)
            j = findfirst(x->x==alpha,GenPoints2)
            k = findfirst(x->x==beta,GenPoints2)
            if j != nothing && k != nothing
                Volume2[j]+=abs(area(Primitive(alpha,alpha1,beta1)))
                Volume2[k]+=abs(area(Primitive(beta,alpha1,beta1)))
            end
        end
        
        Voldiff=[abs(Volume2[l]-Volume1[l]) for l in 1:m2]
        for edge in c.data._Borders
            if GP==VoronoiDelaunay.getgena(edge)
                GPn=VoronoiDelaunay.getgenb(edge)
            else
                GPn=VoronoiDelaunay.getgena(edge)
            end
            indice=findfirst(x->x==GPn,GenPoints1)
            cn=find(tree,GPn)
            vdif=Voldiff[indice]
            vol=cn.data._Volume/scale_fact^2
            rl=[cn.data._ConservativeVariables[j] for j in 1:shape]
            ri=[c.data._ConservativeVariables[j] for j in 1:shape]
            r1l=[cn.data._IntensiveVariables[j] for j in 1:shape]
            r1i=[c.data._IntensiveVariables[j] for j in 1:shape]
            cn.data._ConservativeVariables=(vdif*ri+vol*rl)/(vdif+vol)
            cn.data._IntensiveVariables[1]=(vdif*r1i[1]+vol*r1l[1])/(vol+vdif)
            cn.data._IntensiveVariables[2:3]=(vdif*r1i[1]*r1i[2:3]+vol*r1l[1]*r1l[2:3])/((vol+vdif)*cn.data._IntensiveVariables[1])
            cn.data._IntensiveVariables[4]=(vdif*r1i[4]+vol*r1l[4])/(vol+vdif)
            cn.data._Volume=(vol+vdif)*scale_fact^2
        end
        
        GPT=vector(GP)
        
        if c.data._GeneratingPoint==GP && c.data._GeneratingPoint != Point2D(0.0,0.0)
            c.data=Cell(Point2D(0.0,0.0))
        end
    end     
    #return tree
end


#refines the mesh
function refinement(tree)
    ref=[]
    for c in allleaves(tree)
        cell=c.data
        if cell._GeneratingPoint==Point2D(0.0,0.0) 
            continue
        end
        Vi=cell._Volume
        mi=cell._ConservativeVariables[1]*Vi
        GP=cell._GeneratingPoint
        if mi<1.5*mtilda && Vi<1.7*Vtilda
            continue
        end
        push!(ref,GP)
    end

    for GP in ref
        c=find(tree,GP)
        cell=c.data
        cellnew=copy(cell)
        x=getx(GP)
        y=gety(GP)

        #rate=rand()
        alpha,beta=x+(1/n)/sqrt(2),y+(1/n)/sqrt(2)
        if alpha > max_coord2
            alpha=2*max_coord2-alpha
        elseif alpha < min_coord2
            alpha=2*min_coord2-alpha
        end
        if beta > max_coord2
            beta=2*max_coord2-beta
        elseif beta < min_coord2
            beta=2*min_coord2-beta
        end
        cellnew._GeneratingPoint=Point2D(alpha,beta)
        
        x=getx(cellnew._GeneratingPoint)
        y=gety(cellnew._GeneratingPoint)
        
        cn=tree
        while isleaf(cn)==false
            a=cn.boundary.origin+cn.boundary.widths/2
            bi=[(x>a[1])+1,(y>a[2])+1]
            cn=cn[bi[1],bi[2]]
        end
        if cn.data._GeneratingPoint==Point2D(0.0,0.0)
            cn.data=cellnew
        else
            celln=cn.data
            cn.data=Cell(Point2D(0.0,0.0))
            Ptj=vector(celln._GeneratingPoint)
            split!(cn,[Cell(Point2D(0.0,0.0)),Cell(Point2D(0.0,0.0)),Cell(Point2D(0.0,0.0)),Cell(Point2D(0.0,0.0))])
            a=cn.boundary.origin+cn.boundary.widths/2
            bi=[(x>a[1])+1,(y>a[2])+1]
            bj=[(Ptj[1]>a[1])+1,(Ptj[2]>a[2])+1]
            while bi==bj
                cn=cn[bi[1],bi[2]]
                split!(cn,[Cell(Point2D(0.0,0.0)),Cell(Point2D(0.0,0.0)),Cell(Point2D(0.0,0.0)),Cell(Point2D(0.0,0.0))])
                a=cn.boundary.origin+cn.boundary.widths/2
                bi=[(x>a[1])+1,(y>a[2])+1]
                bj=[(Ptj[1]>a[1])+1,(Ptj[2]>a[2])+1]
            end
            cn[bi[1],bi[2]].data=cellnew
            cn[bj[1],bj[2]].data=celln
        end
    end
    #return tree
end


#update the gradient of active cells
function update_gradients(tree,currentK)
    for c in allleaves(tree)
        GP=c.data._GeneratingPoint
        if c.data._Type == "ghost" || c.data._K < currentK || GP==Point2D(0.,0.)
            continue
        end
        c.data._averagegradient=gradientmoyen(tree,GP)
        
        DELTAij=[[] for l in 1:shape]
        PSIij=[[] for l in 1:shape]
        si=vector(c.data._centreofmass)*scale_fact
        intvar=[c.data._IntensiveVariables[l] for l in 1:shape]
        psimintempo=[intvar[l] for l in 1:shape]
        psimaxtempo=[intvar[l] for l in 1:shape]
        for border in c.data._Borders
            if GP==VoronoiDelaunay.getgena(border)
                GPn=VoronoiDelaunay.getgenb(border)
            else
                GPn=VoronoiDelaunay.getgena(border)
            end
            cn=find(tree,GPn)
            a,b=VoronoiDelaunay.geta(border), VoronoiDelaunay.getb(border)
            centreofface = 0.5*(vector(a)+vector(b))*scale_fact
            psijtempo=cn.data._IntensiveVariables
            for l in 1:shape
                if psijtempo[l]>psimaxtempo[l]
                    psimaxtempo[l]=psijtempo[l]
                end
                if psijtempo[l]<psimintempo[l]
                    psimintempo[l]=psijtempo[l]
                end
                push!(DELTAij[l],scalarproduct2D(c.data._averagegradient[l],centreofface-si))
            end
        end
        for l in 1:shape
            for delta in DELTAij[l]
                push!(PSIij[l],PSImaker(psimaxtempo[l],psimintempo[l],intvar[l],delta))
            end
            alphai=1
            for psi in PSIij[l]
                if psi < alphai
                    alphai=psi
                end
            end
            c.data._averagegradient[l]=alphai*c.data._averagegradient[l]
        end
    end
    return tree
end


#update the Intensive Variables of active cells
function update_IV(tree,currentK)
    for c in allleaves(tree)
        if c.data._GeneratingPoint == Point2D(0.,0.) || c.data._Type == "ghost"
            continue
        end
        if c.data._K >= currentK
            c.data._IntensiveVariables=from_Conservative_to_Intensive(c.data._ConservativeVariables)
            if c.data._IntensiveVariables[4]<0 && abs(c.data._IntensiveVariables[4])<TOL
                c.data._IntensiveVariables[4]=0.
                c.data._ConservativeVariables=from_Intensive_to_Conservative(c.data._IntensiveVariables)
            end
        end
        c.data._SoundSpeed=sqrt(c.data._IntensiveVariables[4]*gamma/c.data._IntensiveVariables[1])
        c.data._Radius=sqrt(c.data._Volume/pi)
    end
    return tree
end

#returns the data from a tree to a mesh structure
function from_tree_to_mesh(tree)
    activecells=Cell{Point2D}[]
    ghostcells=Cell{Point2D}[]
    
    for c in allleaves(tree)
        cell=c.data
        if cell._GeneratingPoint==Point2D(0.,0.)
            continue
        elseif cell._Type == "active"
            push!(activecells,cell)
        else
            push!(ghostcells,cell)
        end
    end
    mesh=Mesh(activecells,ghostcells)
    return mesh
end

#builds the global mesh thanks to the parallel channel stored in r
function mesh_construction(r)
    cells=Cell{Point2D}[]
    for future in r
        mesh=fetch(future)

        cells=vcat(cells,mesh._activecells)
    end
    return Mesh(cells)
end
        

#Finds the next time where cells are active, finds k=currentK such as in B-6.2
function time_update(mesh,whichhalf::Vector{Int64},currentK::Int64)
    Kmax=0
    for cell in mesh._activecells
        if cell._K>Kmax && cell._GeneratingPoint != Point2D(0.,0.)
            Kmax=cell._K
        end
    end
    deltat=0
    if currentK>Kmax && whichhalf != []
        a=whichhalf[Kmax:currentK-1]
        i=1
        while i<=currentK-Kmax
            deltat+=(Dt/2^(Kmax+i))*(1-a[i])
            i+=1
        end
    elseif currentK>Kmax && whichhalf == []
        deltat=Dt/2^currentK
    else
        deltat=Dt/2^Kmax
    end

    if currentK < Kmax && currentK != 0
        a=[0 for i in 1:Kmax-1]
        a[1:currentK-1]=whichhalf
        a[currentK]=1
        whichhalf=a
        currentK=Kmax
    elseif currentK == Kmax && currentK != 0 && whichhalf != []
        i=currentK-1
        while i > 0 && whichhalf[i]==1
            i-=1
        end
        currentK=i
        if i>=2
            whichhalf=whichhalf[1:i-1]
        else
            whichhalf=Int64[]
        end
    elseif currentK>Kmax && Kmax != 0 && whichhalf != []
        currentK=Kmax
        i=currentK-1
        while i > 0 && whichhalf[i]==1
            i-=1
        end
        currentK=i
        if i>=2
            whichhalf=whichhalf[1:i-1]
        else
            whichhalf=Int64[]
        end
    else
        whichhalf=[0 for i in 1:Kmax-1]
        currentK=Kmax
    end
    return (deltat,currentK,whichhalf)
end


#Shift all mesh generating point by \Delta t * w_i
function drift(mesh,deltat::Float64)
    for cell in mesh._activecells
        if cell._GeneratingPoint == Point2D(0.,0.)
            continue
        end
        
        cell._ConservativeVariables+=cell._Ftot/cell._Volume
        cell._Ftot=[0. for k in 1:shape]
        
        alpha,beta=vector(cell._GeneratingPoint)+cell._W*deltat/scale_fact
        if alpha > max_coord2
            alpha=2*max_coord2-alpha
        elseif alpha < min_coord2
            alpha=2*min_coord2-alpha
        end
        if beta > max_coord2
            beta=2*max_coord2-beta
        elseif beta < min_coord2
            beta=2*min_coord2-beta
        end
        cell._GeneratingPoint=Point2D(alpha,beta)
    end
end

#Makes cells rounder
function round(mesh,deltat::Float64)
    for cell in mesh._activecells
        if cell._GeneratingPoint == Point2D(0.,0.)
            continue
        end
        Displacement1=[0.,0.]
        di=(vector(cell._centreofmass)-vector(cell._GeneratingPoint))*scale_fact
        normofdi=norm2D(di)
        Ri=cell._Radius
        ci=cell._SoundSpeed
        if normofdi/(nu*Ri)<0.9
            Displacement1=[0.,0.]
        elseif  normofdi/(nu*Ri)<1.1
            Displacement1=khi*ci*(normofdi-0.9*nu*Ri)*di/(normofdi*0.2*nu*Ri)
        else
            Displacement1=khi*ci*di/normofdi
        end
        alpha,beta=vector(cell._GeneratingPoint)+Displacement1*deltat/scale_fact
        if alpha > max_coord2
            alpha=2*max_coord2-alpha
        elseif alpha < min_coord2
            alpha=2*min_coord2-alpha
        end
        if beta > max_coord2
            beta=2*max_coord2-beta
        elseif beta < min_coord2
            beta=2*min_coord2-beta
        end
        cell._GeneratingPoint=Point2D(alpha,beta)
    end
    return mesh
end

#Makes a domain decomposition for periodic boundary conditions
function decomposition_periodicbc(mesh,nprocs)
    m=Int(sqrt(nprocs))
    
    for cell in mesh._activecells
        x,y=getx(cell._GeneratingPoint),gety(cell._GeneratingPoint)
        X,Y=floor(Int64,m*(x-min_coord2)/width1),floor(Int64,m*(y-min_coord2)/width1)
        process=(X+m*Y) + 2
        cell._process=process
        if x-min_coord2 < marge_max
            cellnew=copy(cell)
            cellnew._GeneratingPoint=Point2D(x-min_coord2+max_coord2,y)
            cellnew._Type = "ghost"
            push!(mesh._ghostcells,cellnew)
            if max_coord2-y < marge_min
                cellnew=copy(cell)
                cellnew._GeneratingPoint=Point2D(x-min_coord2+max_coord2,min_coord2-max_coord2+y)
                cellnew._Type = "ghost"
                push!(mesh._ghostcells,cellnew)
            elseif y-min_coord2 < marge_max
                cellnew=copy(cell)
                cellnew._GeneratingPoint=Point2D(x-min_coord2+max_coord2,y-min_coord2+max_coord2)
                cellnew._Type = "ghost"
                push!(mesh._ghostcells,cellnew)
            end
        end
        if y-min_coord2 < marge_max
            cellnew=copy(cell)
            cellnew._GeneratingPoint=Point2D(x,y-min_coord2+max_coord2)
            cellnew._Type = "ghost"
            push!(mesh._ghostcells,cellnew)
        end
        if max_coord2-x < marge_min
            cellnew=copy(cell)
            cellnew._GeneratingPoint=Point2D(min_coord2-max_coord2+x,y)
            cellnew._Type = "ghost"
            push!(mesh._ghostcells,cellnew)
            if max_coord2-y < marge_min
                cellnew=copy(cell)
                cellnew._GeneratingPoint=Point2D(min_coord2-max_coord2+x,min_coord2-max_coord2+y)
                cellnew._Type = "ghost"
                push!(mesh._ghostcells,cellnew)
            elseif y-min_coord2 < marge_max
                cellnew=copy(cell)
                cellnew._GeneratingPoint=Point2D(min_coord2-max_coord2+x,y-min_coord2+max_coord2)
                cellnew._Type = "ghost"
                push!(mesh._ghostcells,cellnew)
            end
        end
        if max_coord2-y < marge_min
            cellnew=copy(cell)
            cellnew._GeneratingPoint=Point2D(x,min_coord2-max_coord2+y)
            cellnew._Type = "ghost"
            push!(mesh._ghostcells,cellnew)
        end
    end
end

#Makes a domain decomposition for symmetric boundary conditions
function decomposition_symmetricbc(mesh,nprocs)
    m=Int(sqrt(nprocs))
    
    for cell in mesh._activecells
        x,y=getx(cell._GeneratingPoint),gety(cell._GeneratingPoint)
        X,Y=floor(Int64,m*(x-min_coord2)/width1),floor(Int64,m*(y-min_coord2)/width1)
        process=(X+m*Y) + 2
        if nprocs > 1
            cell._process=process
        else
            cell._process = 2
        end
        if x-min_coord2 < marge_min
            cellnew=copy(cell)
            cellnew._GeneratingPoint=Point2D(2*min_coord2-x,y)
            cellnew._IntensiveVariables[2],cellnew._ConservativeVariables[2],cellnew._averagegradient[2]=-cellnew._IntensiveVariables[2],-cellnew._ConservativeVariables[2],-cellnew._averagegradient[2]
            cellnew._Type = "ghost"
            push!(mesh._ghostcells,cellnew)
            if max_coord2-y < marge_max
                cellnew=copy(cell)
                cellnew._GeneratingPoint=Point2D(2*min_coord2-x,2*max_coord2-y)
                cellnew._IntensiveVariables[2:3],cellnew._ConservativeVariables[2:3],cellnew._averagegradient[2:3]=-cellnew._IntensiveVariables[2:3],-cellnew._ConservativeVariables[2:3],-cellnew._averagegradient[2:3]
                cellnew._Type = "ghost"
                push!(mesh._ghostcells,cellnew)
            elseif y-min_coord2 < marge_min
                cellnew=copy(cell)
                cellnew._GeneratingPoint=Point2D(2*min_coord2-x,2*min_coord2-y)
                cellnew._IntensiveVariables[2:3],cellnew._ConservativeVariables[2:3],cellnew._averagegradient[2:3]=-cellnew._IntensiveVariables[2:3],-cellnew._ConservativeVariables[2:3],-cellnew._averagegradient[2:3]
                cellnew._Type = "ghost"
                push!(mesh._ghostcells,cellnew)
            end
        end
        if y-min_coord2 < marge_min
            cellnew=copy(cell)
            cellnew._GeneratingPoint=Point2D(x,2*min_coord2-y)
            cellnew._IntensiveVariables[3],cellnew._ConservativeVariables[3],cellnew._averagegradient[3]=-cellnew._IntensiveVariables[3],-cellnew._ConservativeVariables[3],-cellnew._averagegradient[3]
            cellnew._Type = "ghost"
            push!(mesh._ghostcells,cellnew)
        end
        if max_coord2-x < marge_max
            cellnew=copy(cell)
            cellnew._GeneratingPoint=Point2D(2*max_coord2-x,y)
            cellnew._IntensiveVariables[2],cellnew._ConservativeVariables[2],cellnew._averagegradient[2]=-cellnew._IntensiveVariables[2],-cellnew._ConservativeVariables[2],-cellnew._averagegradient[2]
            cellnew._Type = "ghost"
            push!(mesh._ghostcells,cellnew)
            if max_coord2-y < marge_max
                cellnew=copy(cell)
                cellnew._GeneratingPoint=Point2D(2*max_coord2-x,2*max_coord2-y)
                cellnew._IntensiveVariables[2:3],cellnew._ConservativeVariables[2:3],cellnew._averagegradient[2:3]=-cellnew._IntensiveVariables[2:3],-cellnew._ConservativeVariables[2:3],-cellnew._averagegradient[2:3]
                cellnew._Type = "ghost"
                push!(mesh._ghostcells,cellnew)
            elseif y-min_coord2 < marge_max
                cellnew=copy(cell)
                cellnew._GeneratingPoint=Point2D(2*max_coord2-x,2*min_coord2-y)
                cellnew._IntensiveVariables[2:3],cellnew._ConservativeVariables[2:3],cellnew._averagegradient[2:3]=-cellnew._IntensiveVariables[2:3],-cellnew._ConservativeVariables[2:3],-cellnew._averagegradient[2:3]
                cellnew._Type = "ghost"
                push!(mesh._ghostcells,cellnew)
            end
        end
        if max_coord2-y < marge_max
            cellnew=copy(cell)
            cellnew._GeneratingPoint=Point2D(x,2*max_coord2-y)
            cellnew._IntensiveVariables[3],cellnew._ConservativeVariables[3],cellnew._averagegradient[3]=-cellnew._IntensiveVariables[3],-cellnew._ConservativeVariables[3],-cellnew._averagegradient[3]
            cellnew._Type = "ghost"
            push!(mesh._ghostcells,cellnew)
        end
    end
end


#Builds the mesh structure in each worker processor, stores every cell not belonging to the processor as ghost cells
function from_main_to_worker(mesh,proc)
    cells1=Cell{Point2D}[]
    cells2=Cell{Point2D}[]
    for cell in mesh._activecells
        if cell._process == proc
            push!(cells1,cell)
        else
            cell._Type="ghost"
            push!(cells2,cell)
        end
    end
    for cell in mesh._ghostcells
        if cell._process != proc
            push!(cells2,cell)
        end
    end
    return Mesh(cells1,cells2)
end

#finds ghosts for each mesh by removing all cells not belonging to the processor that do not take part into the computation
function research_ghost(mesh)
    tess=VoronoiDelaunay.DelaunayTessellation()
    GenPoints=Point2D[]
    cells1=mesh._activecells
    cells2=Cell{Point2D}[]
    for cell in mesh._activecells
        push!(GenPoints,cell._GeneratingPoint)
    end
    VoronoiDelaunay.push!(tess,GenPoints)
    size=tess._last_trig_index
    visited = zeros(Bool,size)
    visited[1]=true
    m=length(mesh._ghostcells)
    
    while sum(visited)<size
        GenPoints1=Point2D[]
        l=0
        for ix in 2:size
            trig=tess._trigs[ix]
            na,nb,nc=trig._neighbour_a,trig._neighbour_b,trig._neighbour_c
            bool=(na<ix && na != 1 && !visited[na])||(nb<ix && nb != 1 && !visited[nb])||(nc<ix && nc != 1 && !visited[nc])
            if visited[ix]
                continue
            elseif VoronoiDelaunay.isexternal(trig)
                if findfirst(x->x==geta(trig),GenPoints)!=nothing
                    kmin=0
                    cellcloser=Cell(Point2D(0.,0.))
                    distcloser=10.
                    k=1
                    while k<=m
                        cell=mesh._ghostcells[k]
                        GP=cell._GeneratingPoint
                        dist=norm2D(vector(geta(trig))-vector(GP))
                        if dist<distcloser
                            kmin=k
                            cellcloser=cell
                            distcloser=dist
                        end
                        k+=1
                    end
                    push!(GenPoints1,cellcloser._GeneratingPoint)
                    deleteat!(mesh._ghostcells,kmin)
                    m-=1
                    push!(cells2,cellcloser)
                    visited[na]=false
                    visited[nb]=false
                    visited[nc]=false

                elseif findfirst(x->x==getc(trig),GenPoints)!=nothing
                    kmin=0
                    cellcloser=Cell(Point2D(0.,0.))
                    distcloser=10.
                    k=1
                    while k<=m
                        cell=mesh._ghostcells[k]
                        GP=cell._GeneratingPoint
                        dist=norm2D(vector(getc(trig))-vector(GP))
                        if dist<distcloser
                            kmin=k
                            cellcloser=cell
                            distcloser=dist
                        end
                        k+=1
                    end
                    push!(GenPoints1,cellcloser._GeneratingPoint)
                    deleteat!(mesh._ghostcells,kmin)
                    m-=1
                    push!(cells2,cellcloser)
                    visited[na]=false
                    visited[nb]=false
                    visited[nc]=false

                elseif findfirst(x->x==getb(trig),GenPoints)!=nothing
                    kmin=0
                    cellcloser=Cell(Point2D(0.,0.))
                    distcloser=10.
                    k=1
                    while k<=m
                        cell=mesh._ghostcells[k]
                        GP=cell._GeneratingPoint
                        dist=norm2D(vector(getb(trig))-vector(GP))
                        if dist<distcloser
                            kmin=k
                            cellcloser=cell
                            distcloser=dist
                        end
                        k+=1
                    end
                    push!(GenPoints1,cellcloser._GeneratingPoint)
                    deleteat!(mesh._ghostcells,kmin)
                    m-=1
                    push!(cells2,cellcloser)
                    visited[na]=false
                    visited[nb]=false
                    visited[nc]=false

                elseif !bool
                    visited[ix]=true
                end
                continue
            #elseif VoronoiDelaunay.isexternal(trig) && pale
                #if !bool
                    #visited[ix]=true
                #end
                #continue
            elseif findfirst(x->x==geta(trig),GenPoints)==nothing && findfirst(x->x==getb(trig),GenPoints)==nothing && findfirst(x->x==getc(trig),GenPoints)==nothing
                if !bool
                    visited[ix]=true
                end
                continue
            end
            flag=true
            k=1
            while k<=m
                cell=mesh._ghostcells[k]
                GP=cell._GeneratingPoint
                if incircle(trig,GP) > 0
                    push!(GenPoints1,GP)
                    deleteat!(mesh._ghostcells,k)
                    m-=1
                    push!(cells2,cell)
                    flag=false
                else
                    k+=1
                end
            end
            visited[ix]=flag
            if !flag
                visited[na]=false
                visited[nb]=false
                visited[nc]=false
            end
        end
        VoronoiDelaunay.push!(tess,GenPoints1)
        size1=tess._last_trig_index
        visited1 = zeros(Bool,size1)
        visited[1]=true
        visited1[1:size]=visited
        size=size1
        visited=visited1
    end
    for trig in tess._trigs
        if VoronoiDelaunay.isexternal(trig) && (findfirst(x->x==geta(trig),GenPoints)!=nothing || findfirst(x->x==getb(trig),GenPoints)!=nothing || findfirst(x->x==getc(trig),GenPoints)!=nothing)
            return nothing
        end
    end
    return Mesh(cells1,cells2)
end


## Update ghost cells here (we update : GP,CV,IV,vol,com,SS,R,Deltat,K, les autres on s'en carre)
#Before the loop

function this(tree)
    return tree
end

function ghost_IV_init(r)
    for i in 1:nprocs
        tree = fetch(r[i])
        for c in allleaves(tree)
            GP=c.data._GeneratingPoint
            if GP==Point2D(0.,0.) || c.data._Type == "active"
                continue
            end
            x,y=1.,1.
            if getx(GP)>max_coord2
                x=2*min_coord2-x
            elseif getx(GP)<min_coord2
                x=2*max_coord-x
            end
            if gety(GP)>max_coord2
                y=2*min_coord2-y
            elseif gety(GP)<min_coord2
                y=2*max_coord-y
            end
            GPn=Point2D(x,y)
            proc=c.data._process
            nc=find(fetch(r[proc-1]),GPn)
            if nc==nothing
                continue
            end
            newcell=nc.data
            c.data._IntensiveVariables=newcell._IntensiveVariables
            c.data._ConservativeVariables=newcell._ConservativeVariables
            c.data._Volume=newcell._Volume
            c.data._SoundSpeed=newcell._SoundSpeed
            c.data._Radius=newcell._Radius
            c.data._centreofmass=newcell._centreofmass
            c.data._perimetre=newcell._perimetre
        end
        r[i] = remotecall(this,i+1,tree)
        
    end
end

function ghost_time_init(r)
    Kmax=0
    for i in 1:nprocs
        tree = fetch(r[i])
        for c in allleaves(tree)
            GP=c.data._GeneratingPoint
            if c.data._K > Kmax
                Kmax=c.data._K
            end
            x,y=1.,1.
            if getx(GP)>max_coord2
                x=2*min_coord2-x
            elseif getx(GP)<min_coord2
                x=2*max_coord-x
            end
            if gety(GP)>max_coord2
                y=2*min_coord2-y
            elseif gety(GP)<min_coord2
                y=2*max_coord-y
            end
            GPn=Point2D(x,y)
            if GP==Point2D(0.,0.) || c.data._Type == "active"
                continue
            end
            proc=c.data._process
            nc=find(fetch(r[proc-1]),GPn)
            if nc==nothing
                continue
            end
            newcell=nc.data
            c.data._Deltat=newcell._Deltat
            c.data._K=newcell._K
        end
        r[i] = remotecall(this,i+1,tree)
        
    end
    return Kmax
end
            
     
function ghost_grad_speed(r)        
    for i in 1:nprocs
        tree = fetch(r[i])
        for c in allleaves(tree)
            x,y=1.,1.
            GP=c.data._GeneratingPoint
            if getx(GP)>max_coord2
                x=2*min_coord2-x
            elseif getx(GP)<min_coord2
                x=2*max_coord-x
            end
            if gety(GP)>max_coord2
                y=2*min_coord2-y
            elseif gety(GP)<min_coord2
                y=2*max_coord-y
            end
            GPn=Point2D(x,y)
            if GP==Point2D(0.,0.) || c.data._Type == "active"
                continue
            end
            proc=c.data._process
            nc=find(fetch(r[proc-1]),GPn)
            if nc==nothing
                continue
            end
            newcell=nc.data
            c.data._W=newcell._W
            c.data._averagegradient=newcell._averagegradient
        end
        r[i] = remotecall(this,i+1,tree)
        
    end
end

#during the loop
function ghost_IV(r)
    for i in 1:nprocs
        tree = fetch(r[i])
        
        for c in allleaves(tree)
            x,y=1.,1.
            GP=c.data._GeneratingPoint
            if getx(GP)>max_coord2
                x=2*min_coord2-x
            elseif getx(GP)<min_coord2
                x=2*max_coord-x
            end
            if gety(GP)>max_coord2
                y=2*min_coord2-y
            elseif gety(GP)<min_coord2
                y=2*max_coord-y
            end
            GPn=Point2D(x,y)
            if GP==Point2D(0.,0.) || c.data._Type == "active"
                continue
            end
            
            proc=c.data._process
            nc=find(fetch(r[proc-1]),GPn)
            if nc==nothing
                continue
            end
            newcell=nc.data
            c.data._IntensiveVariables=newcell._IntensiveVariables
            c.data._ConservativeVariables=newcell._ConservativeVariables
            c.data._Volume=newcell._Volume
            c.data._SoundSpeed=newcell._SoundSpeed
            c.data._Radius=newcell._Radius
            c.data._centreofmass=newcell._centreofmass
            c.data._perimetre=newcell._perimetre
            c.data._Deltat=newcell._Deltat
            c.data._K=newcell._K
        end
        r[i] = remotecall(this,i+1,tree)
        
    end
end

#plots the state
function getplot(mesh)
    tess=VoronoiDelaunay.DelaunayTessellation()
    for cell in mesh._activecells
        GP=cell._GeneratingPoint
        VoronoiDelaunay.push!(tess,GP)
    end
    x, y = VoronoiDelaunay.getplotxy(VoronoiDelaunay.voronoiedges(tess))
    
    alpha,beta,groups,group_size = VoronoiDelaunay.getplotcolors(VoronoiDelaunay.voronoicells(tess))
    group=[]
    density=[]
    m=length(groups)
    k=1
    while k<=m
        a=groups[k]
        j=group_size[k]
        cell = Cell(Point2D(0.,0.))
        for c in mesh._activecells
            if c._GeneratingPoint==a
                cell=c
                break
            end
        end
        #rho=0.
        #if cell._SoundSpeed != 0
            #rho = (cell._SoundSpeed-sqrt(cell._IntensiveVariables[2]^2+cell._IntensiveVariables[3]^2))/cell._SoundSpeed
        #elseif sqrt(cell._IntensiveVariables[2]^2+cell._IntensiveVariables[3]^2) !=0
            #rho = -1.
        #else
            #rho= 1.
        #end
        rho=cell._ConservativeVariables[1]
        k+=1
        for l in 1:j
            push!(group,a)
            push!(density,rho)
        end
    end
    return Gadfly.plot(layer(x=x, y=y, Geom.path),
        layer(x=alpha,y=beta,group=group,color=density,Geom.polygon(preserve_order=true, fill=true)),
        Scale.x_continuous(minvalue=1.0, maxvalue=2.0), Scale.y_continuous(minvalue=1.0, maxvalue=2.0))
end



#First parallel work in preparation
function parallel_initial_one(mesh,proc)
    mesh=from_main_to_worker(mesh,proc)
    mesh=research_ghost(mesh)
    tree=build(mesh)
    tree=space_building(tree)
    tree=computeSSRadius(tree)
    return tree
end

#Second parallel work in preparation
function parallel_initial_two(tree,Ktilda)
    tree=computegradients(tree) 
    tree=computespeed(tree,Ktilda)
    return tree
end

#First parallel work inside the loop
function parallel_boucle_one(tree,currentK)
    tree=flux_update(tree,currentK)
    return from_tree_to_mesh(tree)
end

#Second parallel work inside the loop
function parallel_boucle_two(mesh,proc,currentK)
    mesh=from_main_to_worker(mesh,proc)
    mesh=research_ghost(mesh)
    tree=build(mesh)
    tree=space_building(tree)
    #derefinement(tree) 
    #refinement(tree)
    Mesh=from_tree_to_mesh(tree)
    tree=build(Mesh)
    tree=space_building(tree) 
    tree=update_IV(tree,currentK)
    return tree
end

#Third parallel work inside the loop
function parallel_boucle_three(tree,currentK,Ktilda)
    tree=update_gradients(tree,currentK) 
    tree=computespeed(tree,Ktilda)
    return tree
end



end
