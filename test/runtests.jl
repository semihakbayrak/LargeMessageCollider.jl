using LargeMessageCollider, Distributions, Test

@testset "Message collisions" begin
    @test mean(collide(Normal(0,1),Normal(2,2))) == 0.4
end
