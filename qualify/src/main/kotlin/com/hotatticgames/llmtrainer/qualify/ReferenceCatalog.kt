package com.hotatticgames.llmtrainer.qualify

/**
 * Embedded generic size-class catalog (ILLUSTRATIVE geometry, not specific vendor models). Must equal
 * `reference_catalog` in the golden file: a test enforces it. Real catalogs come from the model registry.
 */
object ReferenceCatalog {
    const val JSON: String = """{"catalog_version":1,"description":"ILLUSTRATIVE generic size classes, not specific vendor models.","models":[{"model_id":"ref-0.5b","params_b":0.5,"layers":24,"kv_heads":2,"head_dim":64,"max_context":8192,"weights_mb":{"Q8_0":506.6,"Q6_K":393.4,"Q5_K_M":339.7,"Q4_K_M":289.1,"Q3_K_M":232.5}},
{"model_id":"ref-1b","params_b":1.2,"layers":16,"kv_heads":8,"head_dim":64,"max_context":8192,"weights_mb":{"Q8_0":1215.9,"Q6_K":944.1,"Q5_K_M":815.4,"Q4_K_M":693.8,"Q3_K_M":557.9}},
{"model_id":"ref-3b","params_b":3.2,"layers":28,"kv_heads":8,"head_dim":128,"max_context":8192,"weights_mb":{"Q8_0":3242.5,"Q6_K":2517.7,"Q5_K_M":2174.4,"Q4_K_M":1850.1,"Q3_K_M":1487.7}},
{"model_id":"ref-7b","params_b":7.2,"layers":32,"kv_heads":8,"head_dim":128,"max_context":8192,"weights_mb":{"Q8_0":7295.6,"Q6_K":5664.8,"Q5_K_M":4892.3,"Q4_K_M":4162.8,"Q3_K_M":3347.4}},
{"model_id":"ref-8b","params_b":8.0,"layers":32,"kv_heads":8,"head_dim":128,"max_context":8192,"weights_mb":{"Q8_0":8106.2,"Q6_K":6294.3,"Q5_K_M":5435.9,"Q4_K_M":4625.3,"Q3_K_M":3719.3}},
{"model_id":"ref-13b","params_b":13.0,"layers":40,"kv_heads":8,"head_dim":128,"max_context":8192,"weights_mb":{"Q8_0":13172.6,"Q6_K":10228.2,"Q5_K_M":8833.4,"Q4_K_M":7516.1,"Q3_K_M":6043.9}}],"quants":[{"name":"Q8_0","bits_per_weight":8.5,"quality_penalty":0.3},
{"name":"Q6_K","bits_per_weight":6.6,"quality_penalty":0.6},
{"name":"Q5_K_M","bits_per_weight":5.7,"quality_penalty":1.0},
{"name":"Q4_K_M","bits_per_weight":4.85,"quality_penalty":1.8},
{"name":"Q3_K_M","bits_per_weight":3.9,"quality_penalty":4.0}],"runtimes":[{"name":"llamacpp-android","overhead_mb":300.0,"scratch_mb_per_1k_ctx":16.0,"kv_dtype":"f16"}],"contexts":[2048,4096,8192],"retrieval":[{"index_ram_mb":128.0,"index_storage_mb":256.0,"top_k":4}]}"""
    fun load(): Catalog = Catalog.parse(JSON)
}
