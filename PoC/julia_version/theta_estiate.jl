"""
The conditional probability f(A|B) of event A occurring when event B is confirmed is

f(A) * f(B|A) / f(B)

Therefore, for binary data where the objective variable consists of 0 and 1, the expression using the explanatory variables is

f(βᵢ│yᵢ) = (g(βᵢ) * L(yᵢ│βᵢ))/f(yᵢ)

and this is the target posterior probability for maximum likelihood estimation.


The logarithm does not change the relationship between the two values,

so in order to find the maximum value,

a natural logarithm transformation is applied and then partial differentiation is performed.

∇βᵢ log(f(βᵢ|yᵢ))

= ∇βᵢ log((g(βᵢ) * L(yᵢ|βᵢ)) / f(yᵢ))

= ∇βᵢ log(g(βᵢ)) + ∇βᵢ log(L(yᵢ|βᵢ)) - ∇βᵢ log(f(yᵢ))

= ∇βᵢ log(g(βᵢ)) + ∇βᵢ log(L(yᵢ|βᵢ))

= -βᵢ + ∇βᵢ log(L(yᵢ|βᵢ)) = 0 - (1)

Consider solving this to estimate β.



Since the likelihood ∀β that the entire reaction pattern will be reproduced is the sum of the power powers of the above

pⱼ(βᵢ) ^ (yᵢⱼ) * (1 - pⱼ(βᵢ)) ^ (1 - yᵢⱼ)

L(yᵢ|βᵢ) = ∏_(j=1)^n [pⱼ(βᵢ)^(yᵢⱼ)*(1-pⱼ(βᵢ))^(1-yᵢⱼ)]


The log-likelihood function with natural log transformation is

log L(yᵢ|βᵢ) = ∑_(i=1)^n [yᵢⱼ * log(pⱼ(βᵢ)) + (1-yᵢⱼ) * log(1-pⱼ(βᵢ))]


Let V be the velocity of the inflection point of each model and X be the position of the inflection point of each model.

pⱼ(βᵢ) = 1 / (1 + exp(-D * Vⱼ * (βᵢ - Xⱼ)))


The partial derivative with respect to θ_i of the log-likelihood function is

∇βᵢ log L(yᵢ|βᵢ) = D * ∑_(j=1)^n [Vⱼ * (yᵢⱼ - pⱼ(βᵢ))]


Substitute this into the left side of equation (1),

-βᵢ + D * ∑_(j=1)^n [Vⱼ * (yᵢⱼ - pⱼ(βᵢ))] = 0
"""

using Optim

"""
# Arguments
- `distinctness::Vector{Float64}`  : V. The velocity of the inflection point of each model.
- `rarity::Vector{Float64}`        : X. The position of the inflection point of each model.
- `result::Vector{Vector{Float64}}`: β. binary data where the objective variable consists of 0 and 1.
"""
function logistic_regression_for_two_matrices(distinctness, rarity, result, D=1.701)

  function logistic_equation(param, distinctnessⱼ, rarityⱼ)
    1 / (1 + exp(-D * distinctnessⱼ * (param - rarityⱼ)))
  end

  function maximum_likelihood_estimator(distinctness, rarity, resultᵢ)
    D * sum([distinctness[j] * (resultᵢ[j] - logistic_equation(param, distinctness[j], rarity[j])) for j ∈ axes(distinctness)])
  end

  function param(resultᵢ)
    Optim.minimizer(optimize(β -> maximum_likelihood_estimator(distinctness, rarity, resultᵢ) - β, -5.0, 5.0))
  end

  return [param(resultᵢ) for resultᵢ ∈ result]

end


