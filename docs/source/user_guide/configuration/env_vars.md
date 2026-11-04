# Environment Variables

vllm-ascend uses the following environment variables to configure the system:

On Ascend, environment validation accepts `VLLM_` names registered by either vLLM or
vllm-ascend. Unregistered names still produce a warning, or a `ValueError` when
`--fail-on-environ-validation` is enabled. Validation checks names, not values.

**Note:** Some environment variables are being migrated to `--additional-config` options. These environment variables are still supported during the migration period, and it is recommended to use `--additional-config` for new deployments. See [Additional Configuration](additional_config.md) for details.

{{ include_code('vllm_ascend/envs.py', start_after='begin-env-vars-definition', end_before='end-env-vars-definition') }}
