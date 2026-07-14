export Microphysics2MParams, WarmRainParams2M, P3IceParams

"""
    WarmRainParams2M

Parameters for 2-moment warm rain processes (Seifert-Beheng 2006).

# Fields
- `seifert_beheng::SB`: SB2006 — all warm rain parameters (autoconversion, accretion, etc.)
- `air_properties::AP`: AirProperties — air properties for evaporation
- `condevap::CE`: MM2015 cond-evap relaxation timescale
- `subdep::SD`: MM2015 sub-dep relaxation timescale
"""
@kwdef struct WarmRainParams2M{SB, AP, CE, SD} <: ParametersType
    seifert_beheng::SB
    air_properties::AP
    condevap::CE
    subdep::SD
end
# Construct WarmRainParams2M from a ClimaParams TOML dictionary
WarmRainParams2M(toml_dict::CP.ParamDict; is_limited = true) =
    WarmRainParams2M(;
        seifert_beheng = SB2006(toml_dict; is_limited),
        air_properties = AirProperties(toml_dict),
        condevap = CondEvap2M(toml_dict),
        subdep = SubDep2M(toml_dict),
    )

Base.show(io::IO, mime::MIME"text/plain", x::WarmRainParams2M) =
    ShowMethods.verbose_show_type_and_fields(io, mime, x)

"""
    P3IceParams{N}

Parameters for P3 ice-phase processes with `N` free ice categories.

# Fields
$(DocStringExtensions.FIELDS)

# Constructor

    P3IceParams(toml_dict::CP.ParamDict; is_limited = true, quadrature_order, quad, inp_depletion_model, n_categories)

builds the components:
- `scheme` = [`ParametersP3`](@ref)
- `terminal_velocity` = [`Chen2022VelType`](@ref)
- `cloud_pdf` = [`CloudParticlePDF_SB2006`](@ref)
- `rain_pdf` = [`RainParticlePDF_SB2006`](@ref)
- `ice_nucleation` = [`Frostenberg2023`](@ref)
- `rain_freezing` = [`RainFreezing`](@ref)
- `inter_category` = [`InterCategoryParams`](@ref) when `n_categories > 1`, `nothing` otherwise

# Keyword arguments
- `is_limited`: use limited rain size-distribution parameters. By default, `true`.
- `quadrature_order`: order of the default `Quadrature.GaussLegendre` rule. By default, `6`.
- `quad`: the size-distribution `Quadrature.QuadratureRule`. By default,
  `Quadrature.GaussLegendre(FT, quadrature_order)`. Pass this to use a rule other
  than Gauss-Legendre.
- `inp_depletion_model`: the F23 INP-activation depletion model. By default,
  [`NIceProxyDepletion`](@ref).
- `moments`: the [`MomentClosure`](@ref) selector passed to [`ParametersP3`](@ref)
  (`:two_moment` by default, or `:three_moment`).
- `liquid`: the [`LiquidFractionTreatment`](@ref) selector passed to
  [`ParametersP3`](@ref) (`:none` by default, or `:predicted`).
- `n_categories`: number of free ice categories, between 1 and 4. By default, `1`.
"""
struct P3IceParams{N, P3, VL, PDc, PDr, HET, RF, IC, INPDM, Q} <: ParametersType
    "The core P3 scheme parameters"
    scheme::P3
    "The terminal velocity parameterization"
    terminal_velocity::VL
    "The cloud droplet size distribution"
    cloud_pdf::PDc
    "The rain drop size distribution"
    rain_pdf::PDr
    "The ice nucleation parameters (empirical INP closure)"
    ice_nucleation::HET
    "The rain freezing parameters (Bigg-type immersion freezing)"
    rain_freezing::RF
    "Inter-category interaction parameters ([`InterCategoryParams`](@ref)), or
    `nothing` for a single ice category"
    inter_category::IC
    "Model for F23 INP-activation depletion. Currently only
    [`NIceProxyDepletion`](@ref) (legacy n_ice-as-proxy form) is provided;
    it sets the value subtracted from `INPC(T)/ρ` in the F23 deposition +
    immersion-cap rates. (A prognostic activation-memory model is deferred
    to a follow-up PR.)"
    inp_depletion_model::INPDM
    "Quadrature rule for the size-distribution integrals
    (deposition / sublimation, melting, riming, ice-rain collection,
    sedimentation). See also [`Quadrature.GaussLegendre`](@ref)."
    quad::Q
    function P3IceParams{N}(
        scheme::P3, terminal_velocity::VL, cloud_pdf::PDc, rain_pdf::PDr,
        ice_nucleation::HET, rain_freezing::RF, inter_category::IC,
        inp_depletion_model::INPDM, quad::Q,
    ) where {N, P3, VL, PDc, PDr, HET, RF, IC, INPDM, Q}
        N isa Int && 1 ≤ N ≤ 4 ||
            throw(ArgumentError("P3IceParams supports 1 to 4 ice categories, got $N"))
        (N == 1) == (inter_category === nothing) || throw(
            ArgumentError(
                "`inter_category` must be `nothing` for a single ice category and an `InterCategoryParams` for $N > 1 categories",
            ),
        )
        return new{N, P3, VL, PDc, PDr, HET, RF, IC, INPDM, Q}(
            scheme, terminal_velocity, cloud_pdf, rain_pdf,
            ice_nucleation, rain_freezing, inter_category, inp_depletion_model, quad,
        )
    end
end
Base.show(io::IO, mime::MIME"text/plain", x::P3IceParams) =
    ShowMethods.verbose_show_type_and_fields(io, mime, x)

function P3IceParams(;
    scheme, terminal_velocity, cloud_pdf, rain_pdf, ice_nucleation, rain_freezing,
    inter_category = nothing,
    inp_depletion_model = NIceProxyDepletion(),
    quad = QUAD.GaussLegendre(Float64, 6),
    # `InterCategoryParams` does not carry the category count it was sized for,
    # so it must be paired with an explicit `n_categories`.
    n_categories::Integer = inter_category === nothing ? 1 :
                            throw(ArgumentError("pass `n_categories` explicitly with `inter_category`")),
)
    return P3IceParams{Int(n_categories)}(
        scheme, terminal_velocity, cloud_pdf, rain_pdf,
        ice_nucleation, rain_freezing, inter_category, inp_depletion_model, quad,
    )
end

function P3IceParams(toml_dict::CP.ParamDict;
    is_limited = true,
    quadrature_order = 6,
    quad = QUAD.GaussLegendre(CP.float_type(toml_dict), quadrature_order),
    inp_depletion_model = NIceProxyDepletion(τ_act = 300),
    moments = :two_moment,
    liquid = :none,
    inter_category = nothing,
    n_categories = inter_category === nothing ? 1 :
                   throw(ArgumentError("pass `n_categories` explicitly with `inter_category`")),
)
    icp =
        inter_category === nothing && n_categories > 1 ?
        InterCategoryParams(toml_dict, Val(n_categories)) : inter_category
    return P3IceParams(;
        scheme = ParametersP3(toml_dict; moments, liquid),
        terminal_velocity = Chen2022VelType(toml_dict),
        cloud_pdf = CloudParticlePDF_SB2006(toml_dict),
        rain_pdf = RainParticlePDF_SB2006(toml_dict; is_limited),
        ice_nucleation = Frostenberg2023(toml_dict),
        rain_freezing = RainFreezing(toml_dict),
        inp_depletion_model,
        quad,
        inter_category = icp,
        n_categories,
    )
end

"""
    n_categories(ice::P3IceParams{N})

Number of P3 ice categories represented by `ice`: the leading type parameter `N`.
"""
n_categories(::P3IceParams{N}) where {N} = N

"""

Unified parameter container for 2-moment microphysics.

Supports:
- **Warm rain only** (SB2006): when `ice` is `nothing`
- **Warm rain + P3 ice**: when `ice` is `P3IceParams`

# Fields
- `warm_rain::WR`: WarmRainParams2M — SB2006 parameters + air properties
- `ice::ICE`: P3IceParams or Nothing — optional P3 ice parameters

# Example
```julia
using CloudMicrophysics.Parameters as CMP

# Warm rain only
mp_warm = CMP.Microphysics2MParams(Float64; with_ice = false)

# Warm rain + P3 ice
mp_p3 = CMP.Microphysics2MParams(Float64; with_ice = true)
```
"""
@kwdef struct Microphysics2MParams{WR, ICE} <: ParametersType
    warm_rain::WR
    ice::ICE
end
Base.show(io::IO, mime::MIME"text/plain", x::Microphysics2MParams) =
    ShowMethods.verbose_show_type_and_fields(io, mime, x)

"""
    Microphysics2MParams(toml_dict::CP.ParamDict; with_ice = false, is_limited = true, quadrature_order, quad, inp_depletion_model)

Create a `Microphysics2MParams` object from a ClimaParams TOML dictionary.

# Arguments
- `toml_dict`: ClimaParams parameter dictionary.

# Keyword arguments
- `with_ice`: include P3 ice-phase parameters. By default, `false`.
- `is_limited`: use limited rain size-distribution parameters. By default, `true`.
- `quadrature_order`: order of the default `Quadrature.GaussLegendre` rule passed to
  [`P3IceParams`](@ref) when `with_ice`. By default, `6`.
- `quad`: the size-distribution `Quadrature.QuadratureRule` passed to
  [`P3IceParams`](@ref) when `with_ice`. By default,
  `Quadrature.GaussLegendre(FT, quadrature_order)`.
- `inp_depletion_model`: the F23 INP-activation depletion model passed to
  [`P3IceParams`](@ref) when `with_ice`. By default, [`NIceProxyDepletion`](@ref).
- `moments`: the [`MomentClosure`](@ref) selector passed to [`P3IceParams`](@ref)
  when `with_ice` (`:two_moment` by default, or `:three_moment`).
- `liquid`: the [`LiquidFractionTreatment`](@ref) selector passed to
  [`P3IceParams`](@ref) when `with_ice` (`:none` by default, or `:predicted`).
- `n_categories`: number of free ice categories passed to [`P3IceParams`](@ref)
  when `with_ice`. By default, `1`.
"""
Microphysics2MParams(toml_dict::CP.ParamDict;
    with_ice = false, is_limited = true,
    quadrature_order = 6,
    quad = QUAD.GaussLegendre(CP.float_type(toml_dict), quadrature_order),
    inp_depletion_model = NIceProxyDepletion(τ_act = 300),
    moments = :two_moment,
    liquid = :none,
    n_categories = 1,
) = Microphysics2MParams(;
    # Warm rain parameters (always present)
    warm_rain = WarmRainParams2M(toml_dict; is_limited),
    # Optional ice phase parameters
    ice = with_ice ?
          P3IceParams(toml_dict; is_limited, quad, inp_depletion_model, moments, liquid, n_categories) :
          nothing,
)
