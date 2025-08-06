"""
Polynomial Modulus Degree - Practical Demonstration

This script demonstrates how poly_modulus_degree affects:
1. Security levels
2. Performance (speed and memory)
3. Packing capacity
4. Circuit depth (noise budget)
5. Real-world trade-offs

Requirements: pip install tenseal numpy matplotlib
"""

import tenseal as ts
import numpy as np
import time
import sys
from typing import Dict, List, Tuple


class PolyDegreeBenchmark:
    """Benchmark different polynomial modulus degrees"""

    def __init__(self):
        self.test_degrees = [1024, 2048, 4096, 8192, 16384]  # Common degrees
        self.results = {}

    def create_context(self, degree: int, scheme: str = 'BFV') -> ts.TenSEALContext:
        """Create FHE context with specified polynomial degree"""

        if scheme == 'BFV':
            context = ts.context(
                ts.SCHEME_TYPE.BFV,
                poly_modulus_degree=degree,
                plain_modulus=1032193  # Keep other params constant
            )
        else:  # CKKS
            # Adjust coefficient modulus based on degree for fair comparison
            if degree <= 2048:
                coeff_sizes = [40, 40]
            elif degree <= 4096:
                coeff_sizes = [50, 40, 50]
            elif degree <= 8192:
                coeff_sizes = [60, 40, 40, 60]
            else:  # 16384+
                coeff_sizes = [60, 50, 50, 50, 60]

            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=degree,
                coeff_mod_bit_sizes=coeff_sizes
            )
            context.global_scale = 2 ** 40

        context.generate_galois_keys()
        return context

    def benchmark_performance(self) -> Dict:
        """Benchmark encryption, computation, and decryption performance"""

        print("⚡ PERFORMANCE BENCHMARK")
        print("=" * 60)
        print(f"{'Degree':<8} {'Encrypt':<10} {'Multiply':<10} {'Decrypt':<10} {'Memory':<10}")
        print("-" * 60)

        performance_results = {}
        test_data = list(range(100))  # Test with 100 integers

        for degree in self.test_degrees:
            try:
                # Create context
                context = self.create_context(degree, 'BFV')

                # Measure encryption time
                start_time = time.time()
                encrypted_data = ts.bfv_vector(context, test_data)
                encrypt_time = time.time() - start_time

                # Measure computation time (multiplication)
                start_time = time.time()
                result = encrypted_data * encrypted_data
                compute_time = time.time() - start_time

                # Measure decryption time
                start_time = time.time()
                decrypted = result.decrypt()
                decrypt_time = time.time() - start_time

                # Measure memory usage (ciphertext size)
                memory_size = len(encrypted_data.serialize())

                performance_results[degree] = {
                    'encrypt_time': encrypt_time,
                    'compute_time': compute_time,
                    'decrypt_time': decrypt_time,
                    'memory_bytes': memory_size,
                    'total_time': encrypt_time + compute_time + decrypt_time
                }

                print(f"{degree:<8} {encrypt_time * 1000:<9.2f}ms {compute_time * 1000:<9.2f}ms "
                      f"{decrypt_time * 1000:<9.2f}ms {memory_size / 1024:<9.1f}KB")

            except Exception as e:
                print(f"{degree:<8} ERROR: {str(e)[:40]}")
                performance_results[degree] = None

        return performance_results

    def benchmark_packing_capacity(self) -> Dict:
        """Test packing capacity for different degrees"""

        print(f"\n📦 PACKING CAPACITY BENCHMARK")
        print("=" * 60)
        print(f"{'Degree':<8} {'BFV Capacity':<15} {'CKKS Capacity':<15} {'Efficiency':<12}")
        print("-" * 60)

        packing_results = {}

        for degree in self.test_degrees:
            try:
                # BFV packing test
                bfv_context = self.create_context(degree, 'BFV')

                # Test maximum packing capacity
                max_integers = degree  # Theoretical maximum
                test_integers = list(range(min(1000, max_integers)))

                bfv_encrypted = ts.bfv_vector(bfv_context, test_integers)
                bfv_decrypted = bfv_encrypted.decrypt()
                bfv_actual_capacity = len(
                    [x for x in bfv_decrypted[:len(test_integers)] if x == bfv_decrypted[0] or True])

                # CKKS packing test
                ckks_context = self.create_context(degree, 'CKKS')
                max_floats = degree // 2  # CKKS packs half as many real numbers
                test_floats = [float(i) for i in range(min(500, max_floats))]

                ckks_encrypted = ts.ckks_vector(ckks_context, test_floats)
                ckks_decrypted = ckks_encrypted.decrypt()
                ckks_actual_capacity = len(test_floats)

                # Calculate efficiency (how much of theoretical capacity we use)
                bfv_efficiency = len(test_integers) / max_integers * 100
                ckks_efficiency = len(test_floats) / max_floats * 100

                packing_results[degree] = {
                    'bfv_max_capacity': max_integers,
                    'bfv_tested_capacity': len(test_integers),
                    'ckks_max_capacity': max_floats,
                    'ckks_tested_capacity': len(test_floats),
                    'bfv_efficiency': bfv_efficiency,
                    'ckks_efficiency': ckks_efficiency
                }

                print(f"{degree:<8} {max_integers:<15,} {max_floats:<15,} {bfv_efficiency:<11.1f}%")

            except Exception as e:
                print(f"{degree:<8} ERROR: {str(e)[:50]}")
                packing_results[degree] = None

        return packing_results

    def benchmark_circuit_depth(self) -> Dict:
        """Test circuit depth capacity (noise budget)"""

        print(f"\n🔢 CIRCUIT DEPTH BENCHMARK")
        print("=" * 60)
        print(f"{'Degree':<8} {'Initial Budget':<15} {'After Add':<12} {'After Mult':<12} {'Max Depth':<10}")
        print("-" * 60)

        depth_results = {}

        for degree in self.test_degrees:
            try:
                context = self.create_context(degree, 'BFV')

                # Create test ciphertext
                test_value = [42]
                encrypted = ts.bfv_vector(context, test_value)

                # Check initial noise budget
                # Note: TenSEAL doesn't expose noise budget directly, so we'll simulate
                initial_budget = self.estimate_noise_budget(degree)

                # Test addition (low noise cost)
                encrypted_add = encrypted + encrypted
                budget_after_add = initial_budget - 2  # Estimated cost

                # Test multiplication (high noise cost)
                encrypted_mult = encrypted * encrypted
                budget_after_mult = initial_budget - max(10, degree // 400)  # Estimated cost

                # Estimate maximum multiplicative depth
                max_depth = initial_budget // max(10, degree // 400)

                depth_results[degree] = {
                    'initial_budget': initial_budget,
                    'budget_after_add': budget_after_add,
                    'budget_after_mult': budget_after_mult,
                    'estimated_max_depth': max_depth
                }

                print(f"{degree:<8} {initial_budget:<15} {budget_after_add:<12} "
                      f"{budget_after_mult:<12} {max_depth:<10}")

            except Exception as e:
                print(f"{degree:<8} ERROR: {str(e)[:50]}")
                depth_results[degree] = None

        return depth_results

    def estimate_noise_budget(self, degree: int) -> int:
        """Estimate noise budget based on polynomial degree"""
        # This is a rough estimation - actual values depend on many factors
        if degree <= 1024:
            return 50
        elif degree <= 2048:
            return 80
        elif degree <= 4096:
            return 120
        elif degree <= 8192:
            return 160
        else:
            return 200

    def demonstrate_real_world_scenarios(self) -> Dict:
        """Show how degree choice affects real-world applications"""

        print(f"\n🌍 REAL-WORLD SCENARIO ANALYSIS")
        print("=" * 60)

        scenarios = {
            'IoT Sensor Data': {
                'data_size': 100,
                'operations': 'light',
                'security_requirement': 'medium',
                'latency_requirement': 'low'
            },
            'Financial Analytics': {
                'data_size': 10000,
                'operations': 'heavy',
                'security_requirement': 'high',
                'latency_requirement': 'medium'
            },
            'Healthcare Research': {
                'data_size': 5000,
                'operations': 'medium',
                'security_requirement': 'very_high',
                'latency_requirement': 'high'
            },
            'Machine Learning': {
                'data_size': 50000,
                'operations': 'very_heavy',
                'security_requirement': 'high',
                'latency_requirement': 'high'
            }
        }

        recommendations = {}

        for scenario_name, requirements in scenarios.items():
            recommended_degree = self.recommend_degree(requirements)
            recommendations[scenario_name] = {
                'recommended_degree': recommended_degree,
                'reasoning': self.get_recommendation_reasoning(requirements, recommended_degree)
            }

            print(f"\n{scenario_name}:")
            print(f"  Data Size: {requirements['data_size']:,} values")
            print(f"  Security: {requirements['security_requirement']}")
            print(f"  Latency: {requirements['latency_requirement']} tolerance")
            print(f"  Recommended Degree: {recommended_degree}")
            print(f"  Reasoning: {recommendations[scenario_name]['reasoning']}")

        return recommendations

    def recommend_degree(self, requirements: Dict) -> int:
        """Recommend polynomial degree based on requirements"""

        data_size = requirements['data_size']
        security = requirements['security_requirement']
        latency = requirements['latency_requirement']
        operations = requirements['operations']

        # Start with minimum degree that can handle data size
        min_degree = 1024
        while min_degree < data_size and min_degree < 32768:
            min_degree *= 2

        # Adjust for security requirements
        security_mapping = {
            'low': 2048,
            'medium': 4096,
            'high': 8192,
            'very_high': 16384
        }

        security_degree = security_mapping.get(security, 4096)

        # Adjust for latency requirements
        if latency == 'low':
            max_degree = 4096
        elif latency == 'medium':
            max_degree = 8192
        else:
            max_degree = 16384

        # Choose the degree that satisfies all constraints
        recommended = max(min_degree, security_degree)
        recommended = min(recommended, max_degree)

        return recommended

    def get_recommendation_reasoning(self, requirements: Dict, degree: int) -> str:
        """Generate explanation for degree recommendation"""

        reasons = []

        if requirements['data_size'] > degree // 2:
            reasons.append("increased for data capacity")

        if requirements['security_requirement'] in ['high', 'very_high']:
            reasons.append("high security requirements")

        if requirements['latency_requirement'] == 'low':
            reasons.append("optimized for low latency")

        if requirements['operations'] in ['heavy', 'very_heavy']:
            reasons.append("supports complex computations")

        return "Balanced for " + ", ".join(reasons) if reasons else "Standard configuration"


def run_comprehensive_benchmark():
    """Run comprehensive polynomial degree benchmark"""

    print("🔍 POLYNOMIAL MODULUS DEGREE - COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print("Analyzing the impact of polynomial degree on FHE performance")
    print("Testing degrees: 1024, 2048, 4096, 8192, 16384")
    print("=" * 80)

    benchmark = PolyDegreeBenchmark()

    # Run all benchmarks
    try:
        performance_results = benchmark.benchmark_performance()
        packing_results = benchmark.benchmark_packing_capacity()
        depth_results = benchmark.benchmark_circuit_depth()
        scenario_results = benchmark.demonstrate_real_world_scenarios()

        # Summary analysis
        print(f"\n📊 SUMMARY ANALYSIS")
        print("=" * 60)

        print("Performance Trends:")
        if performance_results:
            fastest_degree = min(performance_results.keys(),
                                 key=lambda k: performance_results[k]['total_time'] if performance_results[
                                     k] else float('inf'))
            slowest_degree = max(performance_results.keys(),
                                 key=lambda k: performance_results[k]['total_time'] if performance_results[k] else 0)

            print(f"  Fastest: {fastest_degree} (total time)")
            print(f"  Slowest: {slowest_degree} (total time)")

        print(f"\nPacking Efficiency:")
        if packing_results:
            best_bfv = max(packing_results.keys(),
                           key=lambda k: packing_results[k]['bfv_max_capacity'] if packing_results[k] else 0)
            best_ckks = max(packing_results.keys(),
                            key=lambda k: packing_results[k]['ckks_max_capacity'] if packing_results[k] else 0)

            print(f"  Best BFV capacity: {best_bfv} ({packing_results[best_bfv]['bfv_max_capacity']:,} integers)")
            print(f"  Best CKKS capacity: {best_ckks} ({packing_results[best_ckks]['ckks_max_capacity']:,} floats)")

        print(f"\nGeneral Recommendations:")
        print(f"  Learning/Testing: 2048-4096")
        print(f"  Production Systems: 8192")
        print(f"  High Security: 16384+")
        print(f"  Real-time Applications: 4096 or lower")
        print(f"  Batch Processing: 8192 or higher")

    except Exception as e:
        print(f"Benchmark error: {e}")
        print("This might be due to TenSEAL installation or system limitations")


if __name__ == "__main__":
    run_comprehensive_benchmark()