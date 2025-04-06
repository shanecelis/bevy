use bevy_math::Vec4;
use bevy_asset::Handle;
use bevy_render::{prelude::Shader, render_resource::{ShaderType, VertexState}};

pub const VIEWPORT_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(0x80a46558e7ae42b6939893ae4ddc3ee6u128);

/// The GPU representation of the uniform data of a [`ColorMaterial`].
#[derive(Clone, Default, ShaderType)]
pub struct ViewportUniform {
    pub viewport: Vec4,
}

/// uses the [`VIEWPORT_SHADER_HANDLE`] to output a
/// ```wgsl
/// struct ViewportVertexOutput {
///     [[builtin(position)]]
///     position: vec4<f32>;
///     [[location(0)]]
///     uv: vec2<f32>;
/// };
/// ```
/// from the vertex shader.
/// The draw call should render one triangle: `render_pass.draw(0..6, 0..1);`
pub fn viewport_shader_vertex_state() -> VertexState {
    VertexState {
        shader: VIEWPORT_SHADER_HANDLE,
        shader_defs: Vec::new(),
        entry_point: "viewport_vertex_shader".into(),
        buffers: Vec::new(),
    }
}
