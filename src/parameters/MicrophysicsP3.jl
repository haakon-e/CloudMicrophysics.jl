export ParametersP3
export MassPowerLaw, AreaPowerLaw, SlopePowerLaw, SlopeConstant, VentilationFactor
export MomentClosure, TwoMoment, ThreeMoment
export LiquidFractionTreatment, NoLiquidFraction, PredictedLiquidFraction
export InterCategoryParams

### ----------------------------- ###
### --- SUB-PARAMETERIZATIONS --- ###
### ----------------------------- ###

"""
    MassPowerLaw{FT}

Parameters for mass(size) relation.

From measurements of mass grown by vapor diffusion and aggregation in midlatitude cirrus
by Brown and Francis (1995) [BrownFrancis1995](@cite)

A part of the [`ParametersP3`](@ref) parameter set.

!!! note
    The `BF1995_mass_coeff_alpha` parameter is provided in units of [`g μm^(-β_va)`]
    but the `α_va` field is stored in SI-like units of [`kg m^(-β_va)`] for
    consistency with the rest of the code.

# Fields
$(DocStringExtensions.FIELDS)
"""
@kwdef struct MassPowerLaw{FT} <: ParametersType
    "Coefficient in mass(size) relation [`kg m^(-β_va)`]"
    α_va::FT
    "Coefficient in mass(size) relation [`-`]"
    β_va::FT
end
function MassPowerLaw(toml_dict::CP.ParamDict)
    name_map = (;
        :BF1995_mass_coeff_alpha => :α_va,
        :BF1995_mass_exponent_beta => :β_va,
    )
    (; β_va) = p = CP.get_parameter_values(toml_dict, name_map, "CloudMicrophysics")
    α_va = p.α_va * 10^(6 * β_va - 3)
    FT = CP.float_type(toml_dict)
    return MassPowerLaw{FT}(; α_va, β_va)
end

"""
    AreaPowerLaw{FT}

Parameters for area(size) relation.

```math
A(D) = γ D^σ
```

where `γ` and `σ` are coefficients in area(size) for ice side plane, column, bullet,
and planar polycrystal aggregates. Values are from Mitchell (1996) [Mitchell1996](@cite)

A part of the [`ParametersP3`](@ref) parameter set.

# Fields
$(DocStringExtensions.FIELDS)
"""
@kwdef struct AreaPowerLaw{FT} <: ParametersType
    "Scale [`μm^(2-σ)`]"
    γ::FT
    "Power [`-`]"
    σ::FT
end
function AreaPowerLaw(toml_dict::CP.ParamDict)
    name_map = (; :M1996_area_coeff_gamma => :γ, :M1996_area_exponent_sigma => :σ)
    params = CP.get_parameter_values(toml_dict, name_map, "CloudMicrophysics")
    FT = CP.float_type(toml_dict)
    return AreaPowerLaw{FT}(; params...)
end

"""
    SlopeLaw

The top-level super-type for slope parameterizations.

See [`SlopePowerLaw`](@ref) and [`SlopeConstant`](@ref) for concrete implementations.
"""
abstract type SlopeLaw <: ParametersType end

"""
    SlopePowerLaw{FT}

Slope parameter μ as a power law in shape parameter λ:

```math
μ(λ) = a λ^b - c
```

and is limited to:

```math
0 ≤ μ ≤ μ_{max}
```

See also Eq. 3 in Morrison and Milbrandt (2015) [MorrisonMilbrandt2015](@cite)

A part of the [`ParametersP3`](@ref) parameter set.

# Fields
$(DocStringExtensions.FIELDS)
"""
@kwdef struct SlopePowerLaw{FT} <: SlopeLaw
    "Scale [`m^b`]"
    a::FT
    "Power [`-`]"
    b::FT
    "Offset [`-`]"
    c::FT
    "Upper limiter [`-`]"
    μ_max::FT
end
function SlopePowerLaw(toml_dict::CP.ParamDict)
    name_map = (;
        :Heymsfield_mu_coeff1 => :a,
        :Heymsfield_mu_coeff2 => :b,
        :Heymsfield_mu_coeff3 => :c,
        :Heymsfield_mu_cutoff => :μ_max,
    )
    params = CP.get_parameter_values(toml_dict, name_map, "CloudMicrophysics")
    return SlopePowerLaw(; params...)
end

"""
    SlopeConstant{FT}

Slope parameter μ as a constant:

```math
μ(λ) = μ_{const}
```

A part of the [`ParametersP3`](@ref) parameter set.

# Fields
$(DocStringExtensions.FIELDS)
"""
@kwdef struct SlopeConstant{FT} <: SlopeLaw
    "Slope parameter μ [`-`]"
    μ::FT
end
function SlopeConstant(toml_dict::CP.ParamDict)
    name_map = (; :P3_constant_slope_parameterization_value => :μ)
    params = CP.get_parameter_values(toml_dict, name_map, "CloudMicrophysics")
    return SlopeConstant(; params...)
end

"""
    VentilationFactor{FT}

Parameters for ventilation factor:

```math
F(D) = a_{v} + b_{v}  N_{Sc}^{1/3} N_{Re}(D)^{1/2}
```
where `N_{Sc}` is the Schmidt number and `N_{Re}(D)` is the Reynolds number for a particle with diameter `D`.

From Seifert and Beheng (2006) [SeifertBeheng2006](@cite),
see also Eq. (13-61) in Pruppacher and Klett (2010) [PruppacherKlett2010](@cite)

A part of the [`ParametersP3`](@ref) parameter set.

# Fields
$(DocStringExtensions.FIELDS)
"""
@kwdef struct VentilationFactor{FT} <: ParametersType
    "Constant coefficient in ventilation factor [`-`]"
    aᵥ::FT
    "Linear coefficient in ventilation factor [`-`]"
    bᵥ::FT
end
function VentilationFactor(toml_dict::CP.ParamDict)
    name_map = (;
        :SB2006_ventilation_factor_coeff_av => :aᵥ,
        :SB2006_ventilation_factor_coeff_bv => :bᵥ,
    )
    params = CP.get_parameter_values(toml_dict, name_map, "CloudMicrophysics")
    return VentilationFactor(; params...)
end

"""
    LocalRimeDensity{FT}
    (ρ′_rim::LocalRimeDensity)(Rᵢ)

Local rime density parameterization based on Cober and List (1993) [CoberList1993](@cite),
Eq. 16 and 17.

Given an instance `ρ′_rim::LocalRimeDensity`, obtain the local rime density 
for a given Rᵢ [m² s⁻¹ °C⁻¹] by calling `ρ′_rim(Rᵢ)`.

The parameterization is given by:

```math
ρ'_{rim} = a + b R_i + c R_i^2, \\quad 1 ≤ R_i ≤ 8,
```
The range is extended to `R_i ≤ 12`, by linearly interpolating between 
`ρ′_rim(8)` and `ρ_ice = 900 kg/m³`. The latter is the solid bulk ice density.

For calculating Rᵢ, see [`compute_local_rime_density`](@ref CloudMicrophysics.P3Scheme.compute_local_rime_density).
"""
@kwdef struct LocalRimeDensity{FT} <: ParametersType
    "Constant coefficient"
    a::FT
    "Linear coefficient"
    b::FT
    "Quadratic coefficient"
    c::FT
    "Density of solid bulk ice [`kg m⁻³`]"
    ρ_ice::FT
end
function LocalRimeDensity(toml_dict::CP.ParamDict)
    name_map = (;
        :CL1993_local_rime_density_constant_coeff => :a,
        :CL1993_local_rime_density_linear_coeff => :b,
        :CL1993_local_rime_density_quadratic_coeff => :c,
        :density_ice_water => :ρ_ice,
    )
    params = CP.get_parameter_values(toml_dict, name_map, "CloudMicrophysics")
    return LocalRimeDensity(; params...)
end
function ((; a, b, c, ρ_ice)::LocalRimeDensity)(Rᵢ)
    Rᵢ = clamp(Rᵢ, 1, 12)  # P3 fortran code, microphy_p3.f90, Line 3315 clamps to 1 ≤ Rᵢ ≤ 12

    # Eq. 17 in Cober and List (1993), in [kg / m³], valid for 1 ≤ Rᵢ ≤ 8
    ρ′_rim_CL93(Rᵢ) = a + b * Rᵢ + c * Rᵢ^2

    ρ′_rim = if Rᵢ ≤ 8
        ρ′_rim_CL93(Rᵢ)
    else
        # following P3 fortran code, microphy_p3.f90, Line 3323
        #   https://github.com/P3-microphysics/P3-microphysics/blob/main/src/microphy_p3.f90#L3323
        # for 8 < Rᵢ ≤ 12, linearly interpolate between ρ′_rim(8) ≡ 611 kg/m³ and ρ_ice = 916.7 kg/m³
        ρ′_rim8 = ρ′_rim_CL93(8)
        f_ρ_ice = (Rᵢ - 8) / (12 - 8)
        (1 - f_ρ_ice) * ρ′_rim8 + f_ρ_ice * ρ_ice  # Linear interpolation beyond 8.
    end
    return ρ′_rim
end

"""
    AspectRatio

Aspect-ratio treatment for the ice terminal-velocity correction. Each subtype
is a functor `(state, D)` returning the multiplicative velocity factor:
`Oblate` returns `cbrt(ϕᵢ(state, D))`, `NoAspectRatio` returns `1`.
The functor methods are defined in `P3Scheme`, where `ϕᵢ` is available.
"""
abstract type AspectRatio end
struct Oblate <: AspectRatio end
struct NoAspectRatio <: AspectRatio end

"""
    MomentClosure

Moment-closure treatment for the ice size distribution. Concrete subtypes are
[`TwoMoment`](@ref) and [`ThreeMoment`](@ref).
"""
abstract type MomentClosure end

"""
    TwoMoment{SL <: SlopeLaw}

Two-moment ice: the shape parameter μ is closed by a [`SlopeLaw`](@ref) fit μ(λ).

# Fields
$(DocStringExtensions.FIELDS)
"""
struct TwoMoment{SL <: SlopeLaw} <: MomentClosure
    "Slope relation, e.g. [`SlopePowerLaw`](@ref) or [`SlopeConstant`](@ref)"
    slope::SL
end

# Numerical size bounds on log(λ) for the P3 shape solves, shared by the
# two- and three-moment closures. `λ ∈ [e², e¹⁷] 1/m`, i.e. mean size
# `1/λ ∈ [0.04 μm, 135 mm]`.
const P3_LOGλ_MIN = 2
const P3_LOGλ_MAX = 17

"""
    reflectivity_number_window(μ_max)

Compute the admissible sixth-moment-to-number ratio window `(zn_lo, zn_hi)` [m⁶]
spanned by the shape bounds `μ ∈ [0, μ_max]` and
`logλ ∈ [P3_LOGλ_MIN, P3_LOGλ_MAX]`, from `Z/N = Γ(μ+7)/Γ(μ+1) · exp(-6 logλ)`
at the window corners. The lower bound is floored by `floatmin`.
"""
function reflectivity_number_window(μ_max::FT) where {FT}
    zn_hi = exp(SF.loggamma(μ_max + 7) - SF.loggamma(μ_max + 1) - 6 * FT(P3_LOGλ_MIN))
    zn_lo = exp(SF.loggamma(FT(7)) - SF.loggamma(FT(1)) - 6 * FT(P3_LOGλ_MAX))
    return (max(zn_lo, floatmin(FT)), zn_hi)
end

"""
    ThreeMoment{FT}

Three-moment ice: the shape parameter μ is diagnosed from the number, mass, and
sixth-moment (Z) content.

# Fields
$(DocStringExtensions.FIELDS)
"""
@kwdef struct ThreeMoment{FT} <: MomentClosure
    "Upper bound on the diagnosed shape parameter μ [`-`]"
    μ_max::FT
    "Shape parameter μ of freshly nucleated or multiplied ice [`-`]"
    μ_init::FT
    "Lower bound on the mean ice particle mass (number adjustment and reflectivity-tendency coefficients) [`kg`]"
    mean_mass_min::FT
    "Upper bound on the mean ice particle mass (number adjustment and reflectivity-tendency coefficients) [`kg`]"
    mean_mass_max::FT
    "Number presence scale for the sixth-moment recovery from the advected variable [`m⁻³`]"
    n_presence::FT
    "Lower bound of the admissible sixth-moment-to-number ratio `Z/N` [`m⁶`]"
    zn_lo::FT = reflectivity_number_window(μ_max)[1]
    "Upper bound of the admissible sixth-moment-to-number ratio `Z/N` [`m⁶`]"
    zn_hi::FT = reflectivity_number_window(μ_max)[2]
end
function ThreeMoment(toml_dict::CP.ParamDict)
    name_map = (;
        :P3_ice_shape_parameter_max => :μ_max,
        :P3_ice_shape_parameter_initial => :μ_init,
        :P3_ice_mean_mass_min => :mean_mass_min,
        :P3_ice_mean_mass_max => :mean_mass_max,
        :P3_ice_number_presence_concentration => :n_presence,
    )
    params = CP.get_parameter_values(toml_dict, name_map, "CloudMicrophysics")
    return ThreeMoment{CP.float_type(toml_dict)}(; params...)
end

"""
    LiquidFractionTreatment

Predicted-liquid-fraction treatment for ice. Concrete subtypes are
[`NoLiquidFraction`](@ref) and [`PredictedLiquidFraction`](@ref).
"""
abstract type LiquidFractionTreatment end

"""
    NoLiquidFraction

Dry-ice P3: no predicted liquid fraction.
"""
struct NoLiquidFraction <: LiquidFractionTreatment end

"""
    PredictedLiquidFraction{FT}

Predicted bulk liquid mass carried on ice.

# Fields
$(DocStringExtensions.FIELDS)
"""
@kwdef struct PredictedLiquidFraction{FT} <: LiquidFractionTreatment
    "Liquid fraction below which the vapor path uses the ice-core branch [`-`]"
    F_dry::FT
    "Width of the vapor-path switch band starting at `F_dry` [`-`]"
    ΔF_switch::FT
    "Liquid fraction above which the particle is dumped to rain [`-`]"
    F_melt::FT
    "Volumetric mass concentration regularising the liquid mass fraction [`kg m⁻³`]"
    q_liq_present::FT
    "Whole-particle size above which liquid is shed from ice [`m`]"
    D_shd_onset::FT
    "Mean diameter of drops shed from ice into rain [`m`]"
    D_shd_drop::FT
    "Relaxation timescale converting the shed-able liquid mass into a rate [`s`]"
    τ_shd::FT
    function PredictedLiquidFraction{FT}(
        F_dry,
        ΔF_switch,
        F_melt,
        q_liq_present,
        D_shd_onset,
        D_shd_drop,
        τ_shd,
    ) where {FT}
        # The vapor-path ramp band is [F_dry, F_dry + ΔF_switch]; keep it strictly
        # inside (0, F_melt) so the ramp weight is zero at F_liq = 0 and saturates
        # below the complete-melt threshold.
        @assert 0 < F_dry
        @assert 0 < ΔF_switch
        @assert F_dry + ΔF_switch < F_melt < 1
        @assert q_liq_present > 0
        @assert D_shd_drop < D_shd_onset
        @assert τ_shd > 0
        return new{FT}(F_dry, ΔF_switch, F_melt, q_liq_present, D_shd_onset, D_shd_drop, τ_shd)
    end
end
function PredictedLiquidFraction(toml_dict::CP.ParamDict)
    name_map = (;
        :P3_liquid_fraction_dry_threshold => :F_dry,
        :P3_liquid_fraction_switch_width => :ΔF_switch,
        :P3_liquid_fraction_complete_melt_threshold => :F_melt,
        :P3_liquid_presence_mass_concentration => :q_liq_present,
        :P3_shedding_onset_diameter => :D_shd_onset,
        :P3_shedding_drop_diameter => :D_shd_drop,
        :P3_shedding_timescale => :τ_shd,
    )
    params = CP.get_parameter_values(toml_dict, name_map, "CloudMicrophysics")
    return PredictedLiquidFraction{CP.float_type(toml_dict)}(; params...)
end

"""
    InterCategoryParams{FT}

Parameters for interactions between distinct P3 ice categories: inter-category
collection, destination selection for newly formed ice, and category merging.
Present only for a multi-category configuration (`nothing` when there is a single
ice category).

The initiation threshold `ΔD_init` is selected at construction from five
category-count-specific ClimaParams keys via `Val(N)`; see
[Milbrandt and Morrison (2016)](@cite MilbrandtMorrison2016).

# Fields
$(DocStringExtensions.FIELDS)
"""
@kwdef struct InterCategoryParams{FT} <: ParametersType
    "Base ice-ice collection efficiency [`-`]"
    E_ii::FT
    "Collector rime mass fraction at which the collection-efficiency shutoff ramp starts [`-`]"
    F_rim_shutoff_start::FT
    "Collector rime mass fraction at which collection is fully shut off [`-`]"
    F_rim_shutoff_end::FT
    "Mean-mass-diameter difference threshold for initiating ice into a separate category [`m`]"
    ΔD_init::FT
    "Mean-mass-diameter difference below which two categories are merged [`m`]"
    ΔD_merge::FT
    "Bulk-density difference below which two categories are merged [`kg m⁻³`]"
    Δρ_merge::FT
end

# Select the initiation threshold for `N` categories from the five ClimaParams
# values via `Val(N)` dispatch; `N ≥ 6` uses the six-category value.
@inline _select_ΔD_init(::Val{2}, d2, d3, d4, d5, d6) = d2
@inline _select_ΔD_init(::Val{3}, d2, d3, d4, d5, d6) = d3
@inline _select_ΔD_init(::Val{4}, d2, d3, d4, d5, d6) = d4
@inline _select_ΔD_init(::Val{5}, d2, d3, d4, d5, d6) = d5
@inline _select_ΔD_init(::Val{N}, d2, d3, d4, d5, d6) where {N} = d6

"""
    InterCategoryParams(toml_dict, ::Val{N})
    InterCategoryParams(toml_dict; n_categories)

Construct [`InterCategoryParams`](@ref) for an `N`-category configuration,
selecting `ΔD_init` for `N` from the five per-category-count ClimaParams keys.
`N` must be at least 2.
"""
function InterCategoryParams(toml_dict::CP.ParamDict, ::Val{N}) where {N}
    N ≥ 2 || throw(ArgumentError("InterCategoryParams requires at least 2 categories, got $N"))
    p = CP.get_parameter_values(
        toml_dict,
        (;
            :P3_intercategory_collection_efficiency => :E_ii,
            :P3_intercategory_rime_shutoff_start => :F_rim_shutoff_start,
            :P3_intercategory_rime_shutoff_end => :F_rim_shutoff_end,
            :P3_category_merge_diameter_difference => :ΔD_merge,
            :P3_category_merge_density_difference => :Δρ_merge,
            :P3_category_initiation_diameter_difference_ncat2 => :d2,
            :P3_category_initiation_diameter_difference_ncat3 => :d3,
            :P3_category_initiation_diameter_difference_ncat4 => :d4,
            :P3_category_initiation_diameter_difference_ncat5 => :d5,
            :P3_category_initiation_diameter_difference_ncat6 => :d6,
        ),
        "CloudMicrophysics",
    )
    ΔD_init = _select_ΔD_init(Val(N), p.d2, p.d3, p.d4, p.d5, p.d6)
    FT = CP.float_type(toml_dict)
    return InterCategoryParams{FT}(;
        E_ii = p.E_ii,
        F_rim_shutoff_start = p.F_rim_shutoff_start,
        F_rim_shutoff_end = p.F_rim_shutoff_end,
        ΔD_init,
        ΔD_merge = p.ΔD_merge,
        Δρ_merge = p.Δρ_merge,
    )
end
InterCategoryParams(toml_dict::CP.ParamDict; n_categories::Int) =
    InterCategoryParams(toml_dict, Val(n_categories))

ShowMethods.field_units(::InterCategoryParams) =
    (; ΔD_init = "m", ΔD_merge = "m", Δρ_merge = "kg m⁻³")

### ----------------------------- ###
### --- TOP-LEVEL CONSTRUCTOR --- ###
### ----------------------------- ###

"""
    ParametersP3

Parameters for P3 bulk microphysics scheme.

From Morrison and Milbrandt (2015) [MorrisonMilbrandt2015](@cite)

# Fields
$(DocStringExtensions.FIELDS)
"""
@kwdef struct ParametersP3{FT, MOM <: MomentClosure, LIQ <: LiquidFractionTreatment, AR <: AspectRatio} <:
              ParametersType
    "Mass-size relation, e.g. [`MassPowerLaw`](@ref)"
    mass::MassPowerLaw{FT}
    "Area-size relation, e.g. [`AreaPowerLaw`](@ref)"
    area::AreaPowerLaw{FT}
    "Moment-closure treatment, a [`MomentClosure`](@ref)"
    moments::MOM
    "Predicted-liquid-fraction treatment, a [`LiquidFractionTreatment`](@ref)"
    liquid::LIQ
    "Ventilation relation, e.g. [`VentilationFactor`](@ref)"
    vent::VentilationFactor{FT}
    "Local rime density, e.g. [`LocalRimeDensity`](@ref)"
    ρ_rim_local::LocalRimeDensity{FT}
    "Wet growth time scale [`s`]"
    τ_wet::FT
    "Cloud ice density [`kg m⁻³`]"
    ρ_i::FT
    "Cloud liquid water density [`kg m⁻³`]"
    ρ_l::FT
    "Water freeze temperature [`K`]"
    T_freeze::FT
    "Interim ceiling on the per-particle ice terminal velocity [`m s⁻¹`]"
    v_term_ice_max::FT
    "Lower bound on the mean ice particle mass [`kg`]"
    mean_mass_min::FT
    "Upper bound on the mean ice particle mass [`kg`]"
    mean_mass_max::FT
    "Terminal-velocity aspect-ratio treatment, an [`AspectRatio`](@ref)"
    aspect_ratio::AR = Oblate()
end

"""
    ParametersP3(toml_dict::CP.ParamDict; [slope_law = :powerlaw], [moments = :two_moment], [liquid = :none], [aspect_ratio = Oblate()])

Create a `ParametersP3` object from a `ClimaParams` TOML dictionary.

# Arguments
- `toml_dict::CP.ParamDict`: A `ClimaParams` TOML dictionary
- `slope_law`: Slope law nested inside a `:two_moment` closure (`:constant` or, by default, `:powerlaw`)
- `moments`: Moment closure (`:two_moment` by default, or `:three_moment`)
- `liquid`: Liquid-fraction treatment (`:none` by default, or `:predicted`)
- `aspect_ratio`: an [`AspectRatio`](@ref); by default, `Oblate()`

"""
function ParametersP3(
    toml_dict::CP.ParamDict;
    slope_law = :powerlaw, moments = :two_moment, liquid = :none, aspect_ratio = Oblate(),
)
    @assert slope_law in (:constant, :powerlaw)
    @assert moments in (:two_moment, :three_moment)
    @assert liquid in (:none, :predicted)
    slope = slope_law == :powerlaw ? SlopePowerLaw(toml_dict) : SlopeConstant(toml_dict)
    mom = moments == :two_moment ? TwoMoment(slope) : ThreeMoment(toml_dict)
    liq = liquid == :none ? NoLiquidFraction() : PredictedLiquidFraction(toml_dict)
    params = CP.get_parameter_values(toml_dict,
        (;
            :density_ice_water => :ρ_i,  # TODO: Use `WaterProperties` struct for ice and liquid water density
            :density_liquid_water => :ρ_l,
            :temperature_water_freeze => :T_freeze,
            :P3_wet_growth_timescale => :τ_wet,
            :P3_max_ice_terminal_velocity => :v_term_ice_max,
            :P3_ice_mean_mass_min => :mean_mass_min,
            :P3_ice_mean_mass_max => :mean_mass_max,
        ), "CloudMicrophysics")
    return ParametersP3(;
        mass = MassPowerLaw(toml_dict),
        area = AreaPowerLaw(toml_dict),
        moments = mom,
        liquid = liq,
        vent = VentilationFactor(toml_dict),
        ρ_rim_local = LocalRimeDensity(toml_dict),
        aspect_ratio,
        params...,
    )
end

### ----------------- ###
### ----- UTILS ----- ###
### ----------------- ###

# Unit annotations for verbose show (used by ShowMethods.verbose_show_type_and_fields)
ShowMethods.field_units(::MassPowerLaw) = (; α_va = "kg m^(-β_va)")
ShowMethods.field_units(::AreaPowerLaw) = (; γ = "μm^(2-σ)")
ShowMethods.field_units(::SlopePowerLaw) = (; a = "m^b")
ShowMethods.field_units(::LocalRimeDensity) = (; ρ_ice = "kg m⁻³")
ShowMethods.field_units(::ParametersP3) = (;
    τ_wet = "s", ρ_i = "kg m⁻³", ρ_l = "kg m⁻³", T_freeze = "K",
    v_term_ice_max = "m s⁻¹", mean_mass_min = "kg", mean_mass_max = "kg",
)
