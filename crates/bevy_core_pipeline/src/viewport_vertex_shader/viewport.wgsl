#define_import_path bevy_core_pipeline::viewport_vertex_shader

// Gives viewport in clip space `(min_x, min_y, max_x, max_y)`
@group(1) @binding(0) var<uniform> viewport: vec4<f32>;

struct ViewportVertexOutput {
    @builtin(position)
    position: vec4<f32>,
    @location(0)
    uv: vec2<f32>,
};

// This vertex shader produces the following UVs, when drawn using indices 0..6:
//
//  0 |  0-----------2
//    |  |        . ´4
//    |  |  a  .´    |
//    |  |  .´  b    |
//  1 |  1´5_________3
//  V +---------------
//    U  0           1
//
// The axes are U and V. The region marked a is the upper-left triangle. The
// region marked b is the bottom-right triangle. The digits in the corners of
// the triangles are the vertex indices.
//
// The UV vectors can be converted to clip-space vertices to cover the entire
// screen with the following code:
//
// ```wgsl
// let clip_position = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
// ```
//
// However, this shader seeks to limit itself to a given viewport, so it uses
// the viewport parameter `(min_x, min_y, max_x, max_y)` given in clip-space to
// specify where the vertices are mapped to, e.g. using `viewport =
// vec4<f32>(-1, -1, 1, 1)` recovers the above expression.
@vertex
fn viewport_vertex_shader(@builtin(vertex_index) vertex_index: u32) -> ViewportVertexOutput {
    let index = (vertex_index & 3u) + (vertex_index >> 2u);
    var uv = vec2<f32>(
        f32(index >> 1u),       // X is second bit, 0 or 1.
        f32(index & 1u)         // Y is first bit, 0 or 1.
        );
    let size = viewport.zw - viewport.xy;
    let clip_position = vec4<f32>(uv * vec2<f32>(size.x, -size.y) + vec2<f32>(viewport.x, viewport.z), 0.0, 1.0);

    return ViewportVertexOutput(clip_position, uv);
}
