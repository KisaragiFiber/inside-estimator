use ndarray::Array1;

/// ロジスティック方程式のパラメータθを推定します。
///
/// # 引数
/// - `inflexion_velocity`: 各resultᵢの変曲点の速度。
/// - `rarity`: 各resultᵢの希少性。
/// - `result`: 各resultᵢの結果。
/// - `d`: ロジスティック方程式のDパラメータ。
/// # 戻り値
/// 各resultᵢの推定されたθの値。
pub fn estimator(
    inflexion_velocity: &Array1<f64>,
    rarity: &Array1<f64>,
    result: &Vec<Array1<f64>>,
    d: f64,
) -> Vec<f64> {
    result.iter().map(|result_i| {
        let (mut low, mut high) = (-1.0, 1.0);
        let epsilon = 1e-6;

        while high - low > epsilon {
            let mid = (low + high) / 2.0;
            let mut cost = 0.0;

            for j in 0..inflexion_velocity.len() {
                let logistic = 1.0 / (1.0 + (-d * inflexion_velocity[j] * (mid - rarity[j])).exp());
                cost += d * inflexion_velocity[j] * (result_i[j] - logistic);
            }

            if cost < epsilon {
                high = mid;
            } else if cost > -epsilon {
                low = mid;
            } else if cost.abs() < epsilon {
                break;
            }
        }

        (low + high) / 2.0
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_estimator() {
        let inflexion_velocity = arr1(&[0.8, 0.8, 0.8]);
        let rarity = arr1(&[-0.5, 1.0, 0.0]);
        let result = vec![
            arr1(&[1.0, 1.0, 1.0]),
            arr1(&[1.0, 0.0, 1.0]),
            arr1(&[1.0, 0.0, 0.0]),
        ];
        let d = 1.701;

        let estimated_params = estimator(&inflexion_velocity, &rarity, &result, d);

        assert_eq!(estimated_params.len(), result.len());
        for &param in &estimated_params {
            assert!(param.is_finite());
        }
        println!("estimated_params: {:?}", estimated_params);
        // => estimated_params: [0.9999995231628418, 0.754509449005127, -0.43817949295043945]
    }
}
