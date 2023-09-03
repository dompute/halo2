use crate::multicore;
use crate::plonk::lookup::prover::Committed;
use crate::plonk::permutation::Argument;
use crate::plonk::{lookup, permutation, AdviceQuery, Any, FixedQuery, InstanceQuery, ProvingKey};
use crate::poly::Basis;
use crate::{
    arithmetic::{eval_polynomial, parallelize, CurveAffine},
    poly::{
        commitment::Params, Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff,
        Polynomial, ProverQuery, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};
use group::prime::PrimeCurve;
use group::{
    ff::{BatchInvert, Field, PrimeField, WithSmallOrderMulGroup},
    Curve,
};
use std::any::TypeId;
use std::convert::TryInto;
use std::num::ParseIntError;
use std::slice;
use std::{
    collections::BTreeMap,
    iter,
    ops::{Index, Mul, MulAssign},
};

use super::{ConstraintSystem, Expression};

/// Return the index in the polynomial of size `isize` after rotation `rot`.
fn get_rotation_idx(idx: usize, rot: i32, rot_scale: i32, isize: i32) -> usize {
    (((idx as i32) + (rot * rot_scale)).rem_euclid(isize)) as usize
}

pub use fam::{
    graph::GraphEvaluator,
    value_source::{Calculation, CalculationInfo, ValueSource},
};

/// Evaluator
#[derive(Clone, Default, Debug)]
pub struct Evaluator<C: CurveAffine> {
    ///  Custom gates evalution
    pub custom_gates: GraphEvaluator<C::Scalar>,
    ///  Lookups evalution
    pub lookups: Vec<GraphEvaluator<C::Scalar>>,
}
/// EvaluationData
#[derive(Default, Debug)]
pub struct EvaluationData<C: CurveAffine> {
    /// Intermediates
    pub intermediates: Vec<C::ScalarExt>,
    /// Rotations
    pub rotations: Vec<usize>,
}

impl<C: CurveAffine> Evaluator<C> {
    /// Creates a new evaluation structure
    pub fn new(cs: &ConstraintSystem<C::ScalarExt>) -> Self {
        let mut ev = Evaluator::default();

        // Custom gates
        let mut parts = Vec::new();
        for gate in cs.gates.iter() {
            parts.extend(
                gate.polynomials()
                    .iter()
                    .map(|poly| add_expression(&mut ev.custom_gates, poly)),
            );
        }
        ev.custom_gates.add_calculation(Calculation::Horner(
            ValueSource::PreviousValue(),
            parts,
            ValueSource::Y(),
        ));

        // Lookups
        for lookup in cs.lookups.iter() {
            let mut graph = GraphEvaluator::default();

            let mut evaluate_lc = |expressions: &Vec<Expression<_>>| {
                let parts = expressions
                    .iter()
                    .map(|expr| add_expression(&mut graph, expr))
                    .collect();
                graph.add_calculation(Calculation::Horner(
                    ValueSource::Constant(0),
                    parts,
                    ValueSource::Theta(),
                ))
            };

            // Input coset
            let compressed_input_coset = evaluate_lc(&lookup.input_expressions);
            // table coset
            let compressed_table_coset = evaluate_lc(&lookup.table_expressions);
            // z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            let right_gamma = graph.add_calculation(Calculation::Add(
                compressed_table_coset,
                ValueSource::Gamma(),
            ));
            let lc = graph.add_calculation(Calculation::Add(
                compressed_input_coset,
                ValueSource::Beta(),
            ));
            graph.add_calculation(Calculation::Mul(lc, right_gamma));

            ev.lookups.push(graph);
        }

        ev
    }

    /// Evaluate h poly
    pub(in crate::plonk) fn evaluate_h(
        &self,
        pk: &ProvingKey<C>,
        advice_polys: &[&[Polynomial<C::ScalarExt, Coeff>]],
        instance_polys: &[&[Polynomial<C::ScalarExt, Coeff>]],
        challenges: &[C::ScalarExt],
        y: C::ScalarExt,
        beta: C::ScalarExt,
        gamma: C::ScalarExt,
        theta: C::ScalarExt,
        lookups: &[Vec<lookup::prover::Committed<C>>],
        permutations: &[permutation::prover::Committed<C>],
    ) -> Polynomial<C::ScalarExt, ExtendedLagrangeCoeff> {
        let domain = &pk.vk.domain;
        let size = domain.extended_len();
        let rot_scale = 1 << (domain.extended_k() - domain.k());
        let fixed = &pk.fixed_cosets[..];
        let extended_omega = domain.get_extended_omega();
        let isize = size as i32;
        let one = C::ScalarExt::ONE;
        let l0 = &pk.l0;
        let l_last = &pk.l_last;
        let l_active_row = &pk.l_active_row;
        let p = &pk.vk.cs.permutation;

        // Calculate the advice and instance cosets
        let advice: Vec<Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>> = advice_polys
            .iter()
            .map(|advice_polys| {
                advice_polys
                    .iter()
                    .map(|poly| domain.coeff_to_extended(poly.clone()))
                    .collect()
            })
            .collect();
        let instance: Vec<Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>> = instance_polys
            .iter()
            .map(|instance_polys| {
                instance_polys
                    .iter()
                    .map(|poly| domain.coeff_to_extended(poly.clone()))
                    .collect()
            })
            .collect();

        let mut values = domain.empty_extended();

        // Core expression evaluations
        for (((advice, instance), lookups), permutation) in advice
            .iter()
            .zip(instance.iter())
            .zip(lookups.iter())
            .zip(permutations.iter())
        {
            // Custom gates
            self.custom_gates.evaluate(
                &mut values,
                fixed,
                advice,
                instance,
                &challenges,
                &beta,
                &gamma,
                &theta,
                &y,
                rot_scale,
                isize,
            );

            // Permutations
            let sets = &permutation.sets;
            if !sets.is_empty() {
                let blinding_factors = pk.vk.cs.blinding_factors();
                let last_rotation = Rotation(-((blinding_factors + 1) as i32));
                let chunk_len = pk.vk.cs.degree() - 2;
                let delta_start = beta * &C::Scalar::ZETA;

                let first_set = sets.first().unwrap();
                let last_set = sets.last().unwrap();

                // Permutation constraints
                parallelize(&mut values, |values, start| {
                    let mut beta_term = extended_omega.pow_vartime(&[start as u64, 0, 0, 0]);
                    for (i, value) in values.iter_mut().enumerate() {
                        let idx = start + i;
                        let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                        let r_last = get_rotation_idx(idx, last_rotation.0, rot_scale, isize);

                        // Enforce only for the first set.
                        // l_0(X) * (1 - z_0(X)) = 0
                        *value = *value * y
                            + ((one - first_set.permutation_product_coset[idx]) * l0[idx]);
                        // Enforce only for the last set.
                        // l_last(X) * (z_l(X)^2 - z_l(X)) = 0
                        *value = *value * y
                            + ((last_set.permutation_product_coset[idx]
                                * last_set.permutation_product_coset[idx]
                                - last_set.permutation_product_coset[idx])
                                * l_last[idx]);
                        // Except for the first set, enforce.
                        // l_0(X) * (z_i(X) - z_{i-1}(\omega^(last) X)) = 0
                        for (set_idx, set) in sets.iter().enumerate() {
                            if set_idx != 0 {
                                *value = *value * y
                                    + ((set.permutation_product_coset[idx]
                                        - permutation.sets[set_idx - 1].permutation_product_coset
                                            [r_last])
                                        * l0[idx]);
                            }
                        }
                        // And for all the sets we enforce:
                        // (1 - (l_last(X) + l_blind(X))) * (
                        //   z_i(\omega X) \prod_j (p(X) + \beta s_j(X) + \gamma)
                        // - z_i(X) \prod_j (p(X) + \delta^j \beta X + \gamma)
                        // )
                        let mut current_delta = delta_start * beta_term;
                        for ((set, columns), cosets) in sets
                            .iter()
                            .zip(p.columns.chunks(chunk_len))
                            .zip(pk.permutation.cosets.chunks(chunk_len))
                        {
                            let mut left = set.permutation_product_coset[r_next];
                            for (values, permutation) in columns
                                .iter()
                                .map(|&column| match column.column_type() {
                                    Any::Advice(_) => &advice[column.index()],
                                    Any::Fixed => &fixed[column.index()],
                                    Any::Instance => &instance[column.index()],
                                })
                                .zip(cosets.iter())
                            {
                                left *= values[idx] + beta * permutation[idx] + gamma;
                            }

                            let mut right = set.permutation_product_coset[idx];
                            for values in columns.iter().map(|&column| match column.column_type() {
                                Any::Advice(_) => &advice[column.index()],
                                Any::Fixed => &fixed[column.index()],
                                Any::Instance => &instance[column.index()],
                            }) {
                                right *= values[idx] + current_delta + gamma;
                                current_delta *= &C::Scalar::DELTA;
                            }

                            *value = *value * y + ((left - right) * l_active_row[idx]);
                        }
                        beta_term *= &extended_omega;
                    }
                });
            }

            // Lookups
            for (n, lookup) in lookups.iter().enumerate() {
                // Polynomials required for this lookup.
                // Calculated here so these only have to be kept in memory for the short time
                // they are actually needed.
                let product_coset = pk.vk.domain.coeff_to_extended(lookup.product_poly.clone());
                let permuted_input_coset = pk
                    .vk
                    .domain
                    .coeff_to_extended(lookup.permuted_input_poly.clone());
                let permuted_table_coset = pk
                    .vk
                    .domain
                    .coeff_to_extended(lookup.permuted_table_poly.clone());

                let mut table_values = vec![C::ScalarExt::ZERO; values.len()];
                self.lookups[n].evaluate(
                    &mut table_values,
                    fixed,
                    advice,
                    instance,
                    &challenges,
                    &beta,
                    &gamma,
                    &theta,
                    &y,
                    rot_scale,
                    isize,
                );
                // Lookup constraints
                parallelize(&mut values, |values, start| {
                    for (i, value) in values.iter_mut().enumerate() {
                        let idx = start + i;
                        let table_value = table_values[idx];
                        let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                        let r_prev = get_rotation_idx(idx, -1, rot_scale, isize);

                        let a_minus_s = permuted_input_coset[idx] - permuted_table_coset[idx];
                        // l_0(X) * (1 - z(X)) = 0
                        *value = *value * y + ((one - product_coset[idx]) * l0[idx]);
                        // l_last(X) * (z(X)^2 - z(X)) = 0
                        *value = *value * y
                            + ((product_coset[idx] * product_coset[idx] - product_coset[idx])
                                * l_last[idx]);
                        // (1 - (l_last(X) + l_blind(X))) * (
                        //   z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
                        //   - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta)
                        //          (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
                        // ) = 0
                        *value = *value * y
                            + ((product_coset[r_next]
                                * (permuted_input_coset[idx] + beta)
                                * (permuted_table_coset[idx] + gamma)
                                - product_coset[idx] * table_value)
                                * l_active_row[idx]);
                        // Check that the first values in the permuted input expression and permuted
                        // fixed expression are the same.
                        // l_0(X) * (a'(X) - s'(X)) = 0
                        *value = *value * y + (a_minus_s * l0[idx]);
                        // Check that each value in the permuted lookup input expression is either
                        // equal to the value above it, or the value at the same index in the
                        // permuted table expression.
                        // (1 - (l_last + l_blind)) * (a′(X) − s′(X))⋅(a′(X) − a′(\omega^{-1} X)) = 0
                        *value = *value * y
                            + (a_minus_s
                                * (permuted_input_coset[idx] - permuted_input_coset[r_prev])
                                * l_active_row[idx]);
                    }
                });
            }
        }
        values
    }
}

/// Generates an optimized evaluation for the expression
fn add_expression<F: PrimeField>(
    graph: &mut GraphEvaluator<F>,
    expr: &Expression<F>,
) -> ValueSource {
    match expr {
        Expression::Constant(scalar) => graph.add_constant(scalar),
        Expression::Selector(_selector) => unreachable!(),
        Expression::Fixed(query) => {
            let rot_idx = graph.add_rotation(&query.rotation);
            graph.add_calculation(Calculation::Store(ValueSource::Fixed(
                query.column_index,
                rot_idx,
            )))
        }
        Expression::Advice(query) => {
            let rot_idx = graph.add_rotation(&query.rotation);
            graph.add_calculation(Calculation::Store(ValueSource::Advice(
                query.column_index,
                rot_idx,
            )))
        }
        Expression::Instance(query) => {
            let rot_idx = graph.add_rotation(&query.rotation);
            graph.add_calculation(Calculation::Store(ValueSource::Instance(
                query.column_index,
                rot_idx,
            )))
        }
        Expression::Challenge(challenge) => graph.add_calculation(Calculation::Store(
            ValueSource::Challenge(challenge.index()),
        )),
        Expression::Negated(a) => match **a {
            Expression::Constant(scalar) => graph.add_constant(&-scalar),
            _ => {
                let result_a = add_expression(graph, a);
                match result_a {
                    ValueSource::Constant(0) => result_a,
                    _ => graph.add_calculation(Calculation::Negate(result_a)),
                }
            }
        },
        Expression::Sum(a, b) => {
            // Undo subtraction stored as a + (-b) in expressions
            match &**b {
                Expression::Negated(b_int) => {
                    let result_a = add_expression(graph, a);
                    let result_b = add_expression(graph, b_int);
                    if result_a == ValueSource::Constant(0) {
                        graph.add_calculation(Calculation::Negate(result_b))
                    } else if result_b == ValueSource::Constant(0) {
                        result_a
                    } else {
                        graph.add_calculation(Calculation::Sub(result_a, result_b))
                    }
                }
                _ => {
                    let result_a = add_expression(graph, a);
                    let result_b = add_expression(graph, b);
                    if result_a == ValueSource::Constant(0) {
                        result_b
                    } else if result_b == ValueSource::Constant(0) {
                        result_a
                    } else if result_a <= result_b {
                        graph.add_calculation(Calculation::Add(result_a, result_b))
                    } else {
                        graph.add_calculation(Calculation::Add(result_b, result_a))
                    }
                }
            }
        }
        Expression::Product(a, b) => {
            let result_a = add_expression(graph, a);
            let result_b = add_expression(graph, b);
            if result_a == ValueSource::Constant(0) || result_b == ValueSource::Constant(0) {
                ValueSource::Constant(0)
            } else if result_a == ValueSource::Constant(1) {
                result_b
            } else if result_b == ValueSource::Constant(1) {
                result_a
            } else if result_a == ValueSource::Constant(2) {
                graph.add_calculation(Calculation::Double(result_b))
            } else if result_b == ValueSource::Constant(2) {
                graph.add_calculation(Calculation::Double(result_a))
            } else if result_a == result_b {
                graph.add_calculation(Calculation::Square(result_a))
            } else if result_a <= result_b {
                graph.add_calculation(Calculation::Mul(result_a, result_b))
            } else {
                graph.add_calculation(Calculation::Mul(result_b, result_a))
            }
        }
        Expression::Scaled(a, f) => {
            if *f == F::ZERO {
                ValueSource::Constant(0)
            } else if *f == F::ONE {
                add_expression(graph, a)
            } else {
                let cst = graph.add_constant(f);
                let result_a = add_expression(graph, a);
                graph.add_calculation(Calculation::Mul(result_a, cst))
            }
        }
    }
}

/// Creates a new evaluation structure
// pub fn instance<C: CurveAffine>(graph: &GraphEvaluator<C::ScalarExt>) -> EvaluationData<C> {
//     EvaluationData {
//         intermediates: vec![C::ScalarExt::ZERO; graph.num_intermediates],
//         rotations: vec![0usize; graph.rotations.len()],
//     }
// }

/// Simple evaluation of an expression
pub fn evaluate<F: Field, B: Basis>(
    expression: &Expression<F>,
    size: usize,
    rot_scale: i32,
    fixed: &[Polynomial<F, B>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    challenges: &[F],
) -> Vec<F> {
    let mut values = vec![F::ZERO; size];
    let isize = size as i32;
    parallelize(&mut values, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = start + i;
            *value = expression.evaluate(
                &|scalar| scalar,
                &|_| panic!("virtual selectors are removed during optimization"),
                &|query| {
                    fixed[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|query| {
                    advice[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|query| {
                    instance[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|challenge| challenges[challenge.index()],
                &|a| -a,
                &|a, b| a + &b,
                &|a, b| a * b,
                &|a, scalar| a * scalar,
            );
        }
    });
    values
}
