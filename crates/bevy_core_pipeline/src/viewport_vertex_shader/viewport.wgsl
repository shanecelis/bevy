#define_import_path bevy_core_pipeline::viewport_vertex_shader

// Gives viewport in clip space [x, y, width, height]
@group(1) @binding(0) var<uniform> viewport: vec4<f32>;

struct ViewportVertexOutput {
    @builtin(position)
    position: vec4<f32>,
    @location(0)
    uv: vec2<f32>,
};

// This vertex shader produces the following, when drawn using indices 0..6:
//
//  1 |  0-----x.....2
//  0 |  |  s  |  . ´4
// -1 |  x_____x´
// -2 |  :  .´
// -3 |  1´5         3
//    +---------------
//      -1  0  1  2  3
//
// The axes are clip-space x and y. The region marked s is the visible region.
// The digits in the corners of the right-angled triangle are the vertex
// indices.
//
// The top-left has UV 0,0, the bottom-left has 0,2, and the top-right has 2,0.
// This means that the UV gets interpolated to 1,1 at the bottom-right corner
// of the clip-space rectangle that is at 1,-1 in clip space.
@vertex
fn viewport_vertex_shader(@builtin(vertex_index) vertex_index: u32) -> ViewportVertexOutput {
    // See the explanation above for how this works
    let index = (vertex_index & 3u) + (vertex_index >> 2u);
    var uv = vec2<f32>(
        f32(index >> 1u),       // X: 0 or 1
        f32(index & 1u)         // Y: 0 or 1
        );
    let clip_position = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);

    return ViewportVertexOutput(clip_position, uv);
}
